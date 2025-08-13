#![allow(dead_code)]
#![allow(unused)]

use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;

use crate::model::*;
use crate::model::PVModel;
use crate::replay_buffer::{BoardData, ReplayBuffer, Sampler};
use crate::self_play::ModuleShelf;

use rand::rngs::ThreadRng;
use serde::Deserialize;
#[cfg(feature = "torch")]
use tch::nn::{self, VarStore, Variables, Path};
#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};
#[cfg(feature = "torch")]
use tch::nn::{Optimizer, OptimizerConfig};

use color_eyre::eyre::{self, eyre, Context, ErrReport, OptionExt, Result};
use rand::{prelude, thread_rng, Rng};


#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "type")]
pub enum SumOrAve {
    Sum,
    Ave,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct ModelSyncConfig{
    grad: SumOrAve
}

impl ModelSyncConfig {
    pub fn new(grad: SumOrAve) -> Self {
        Self { grad }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LossValue {
    pub total: f64,
    pub cross_entropy: f64,
    pub mse: f64,
}

impl std::fmt::Display for LossValue{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ce: {}, mse: {}, tot: {} ", self.cross_entropy, self.mse, self.total)
    }
}

impl LossValue {
    fn loss_value(total: f64, cross_entropy: f64, mse: f64) -> Self {
        Self { total, cross_entropy, mse }
    }
}

fn sync_grads<V: BorrowMut<VarStore>>(vss: &mut [V], names: &[String], config: ModelSyncConfig) -> Result<()> {

    let n_model = vss.len();
    if n_model <= 1 {
        return Err(ErrReport::msg("no need to sync"));
    }
    let device = Device::Cpu;

    for name in names.iter() {

        let mut vs_iter = vss.iter();
        let mut sum = vs_iter.next().unwrap()
            .borrow()
            .variables()
            .get(name)
            .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
            .grad()
            .copy()
            .to(device);

        for vs in vs_iter{
            sum += vs
                .borrow()
                .variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .grad()
                .copy()
                .to(device);
        }
        
        let denom = tch::Scalar::float(n_model as f64);
        let reduced_grad = match config.grad {
            SumOrAve::Ave => {
                sum.divide_scalar(denom)
            }
            SumOrAve::Sum => {
                sum
            }
        };

        for vs in vss.iter_mut(){

            let borrow: &mut VarStore = vs.borrow_mut();
            let value = reduced_grad.to_device(borrow.device());
            borrow.variables()
                .get_mut(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .grad()
                .copy_(&value);

        }
    }
    Ok(())
}

// The paper from facebook (https://arxiv.org/abs/1706.02677) claims that the syncing of 
// mean and variance estimate per single mini-batch is redundant and results in the inconsistent loss functions across the workers.
// This point is quite the contrary to the official DDP implementation by pytorch (SyncBatchNorm),
// and chatGPT seems to strongly side with the latter (I tried to 'convince' it multiple times).
// What seems to be the common consensus is that Batch-Norm can actually degrade the sample quality if used on exceptionally small (2 ~ 8) mini-batches.
// syncing the statistics per mini-batch across workers is not ergonomic with the tch-rs framework,
// at least not while using the standard API (one probably has to create custom SyncBatchNorm struct for that from scratch).
// In conclusion, I have currently no other way than only syncing the running stats across workers at the end of one epoch, not per mini-batch.
// Whether 32 samples per worker (as used by AlphaZero) is sufficient to justify this simplification is debatable,
// and I might have to observe the actual statistics of each neuron activation.
// On the other hand, I might be better off using other normalization technique than BN, such as Group N or Layer N.
// Using Layer Normalization will remove the need for this additional sync completely.
fn sync_bn_stats<V: BorrowMut<VarStore>>(vss: &mut [V], names: &[String]) -> Result<()> {

    let n_model = vss.len();
    if n_model <= 1 {
        return Err(ErrReport::msg("no need to sync"));
    }
    let device = Device::Cpu;

    for name in names.iter() {

        let mut vs_iter = vss.iter();
        let mut sum = vs_iter.next().unwrap()
            .borrow()
            .variables()
            .get(name)
            .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
            .copy()
            .to(device);

        for vs in vs_iter{
            sum += vs
                .borrow()
                .variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .copy()
                .to(device);
        }

        let denom = tch::Scalar::float(n_model as f64);
        let mut ave = sum.divide_scalar(denom);

        for vs in vss.iter_mut(){

            let borrow: &mut VarStore = vs.borrow_mut();
            let value = ave.to_device(borrow.device());
            borrow.variables()
                .get_mut(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .copy_(&value);

        }
    }
    Ok(())

}

fn get_trainable_var_names<V: Borrow<VarStore>>(vss: &[V]) -> Result<Vec<String>> {

    let var_store = vss[0].borrow();
    let len = var_store.len();
    if !vss.iter().all(|vs| vs.borrow().len() == len) {
        return Err(eyre!("input VarStores don't share the same length"));
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("weight") || key.contains("bias") {

            for vs in vss.iter(){

                let borrow: &VarStore = vs.borrow();
                if !borrow
                    .variables()
                    .get(key.as_str())
                    .ok_or(eyre!("variable {} does not exist in one of the VarStores", key))?
                    .requires_grad() {
                        return Err(eyre!("Variable {} does not track its gradient!", key));
                    }

            }
            names.push(key.clone());

        }
    }

    Ok(names)
}

fn get_bn_mean_names<V: Borrow<VarStore>>(vss: &[V]) -> Option<Vec<String>> {

    let var_store = vss[0].borrow();
    let len = var_store.len();
    if !vss.iter().all(|vs| vs.borrow().len() == len) {
        return None;
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("running_mean") {

            for vs in vss.iter(){
                let borrow: &VarStore = vs.borrow();
                if None == borrow.variables().get(key.as_str()) {
                    return None;
                } 
            }
            names.push(key.clone());

        }
    }

    Some(names)

}

fn get_bn_var_names<V: Borrow<VarStore>>(vss: &[V]) -> Option<Vec<String>> {

    let var_store = vss[0].borrow();
    let len = var_store.len();
    if !vss.iter().all(|vs| vs.borrow().len() == len) {
        return None;
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("running_var") {

            for vs in vss.iter(){
                let borrow: &VarStore = vs.borrow();
                if None == vs.borrow().variables().get(key.as_str()) {
                    return None;
                } 
            }
            names.push(key.clone());
        }
    }

    Some(names)

}

pub struct NewTrainer<P: PVModel + Send> {
    pub shelf: ModuleShelf<P, P>,
    pub step_count: usize,
    pub loss_record: Vec<LossValue>,
    optimizers: Vec<Optimizer>,
    mini_batch_size: usize,
    trainable_var_names: Vec<String>,
    bn_stats_names: Vec<String>,
    sync_config: ModelSyncConfig,
}

impl<P: PVModel+ Send> NewTrainer<P> {

    pub fn replicas(&self) -> &[(P, VarStore)] {
        &self.shelf.get_group(0).unwrap()[..]
    }

    pub fn replicas_mut(&mut self) -> &[(P, VarStore)] {
        &mut self.shelf.get_group_mut(0).unwrap()[..]
    }

    pub fn new<O: OptimizerConfig + Clone>(
        shelf: ModuleShelf<P, P>,
        config: O,
        lr: f64,
        weight_decay: f64,
        mini_batch_size: usize,
        sync_config: ModelSyncConfig
    ) -> Result<Self>{

        if shelf.table.len() != 1 {
            return Err(eyre!("Trainer creation expected one model group in the shelf but got {}", shelf.table.len()));
        }
        if shelf.table[0].len() == 0 {
            return Err(eyre!("modules has to have at least one element"));
        }

        let optimizers= shelf.table[0].iter().map(|(module, vs)| {

            if let Ok(mut optim) = config.clone().build(vs, lr) {
                optim.set_weight_decay_group(NO_WEIGHT_DECAY_GROUP, weight_decay);
                Ok(optim)
            } else {
                Err(eyre!("Failed to build OptimConfig"))
            }
        }).collect::<Result<Vec<_>>>()?;

        let vss = shelf.table[0].iter().map(|(_, vs)| {
            vs
        }).collect::<Vec<_>>();
        // Get variable names in the nn weight for whom to sync the grads
        let trainable_var_names = get_trainable_var_names(&vss)?;

        // Get batch norm layer stats names
        let mut bn_stats_names = get_bn_mean_names(&vss).ok_or_eyre("Could not get batch norm mean names")?;
        bn_stats_names.extend(get_bn_var_names(&vss).ok_or_eyre("Could not get batch norm var names")?);

        let loss_record: Vec<LossValue> = vec![];
        
        Ok(Self {
            shelf,
            optimizers,
            mini_batch_size,
            trainable_var_names,
            bn_stats_names,
            sync_config,
            loss_record,
            step_count: 0
        })
    }

    // Sample from the replay_buffer, calculate the forward pass on each replica and return the average loss value
    fn forward_backward_pass<D: BoardData, R>(&mut self, replay_buffer: &ReplayBuffer<D>, rng: &mut R) -> LossValue 
    where
    ReplayBuffer<D>: Sampler,
    R: Rng
    {

        let mut cross_entropy_loss = 0.0f64;
        let mut mse_loss = 0.0f64;
        let mut total_loss = 0.0f64;

        for i in 0..self.replicas().len() {

            let (replica, _) = &self.replicas()[i];
            // sample mini-batch from the replay buffer and do the forward pass
            let device = replica.device();
            let (mut position, mut policy, mut reward) 
                = replay_buffer.sample_batch(self.mini_batch_size, (Kind::Float, Device::Cpu), rng).unwrap();
            position = position.to_device(device);
            policy = policy.to_device(device);
            reward = reward.to_device(device);
            let (evaluated_logits, evaluated_reward) = replica.evaluate_t(&position, true);
            let cross_entropy = evaluated_logits.cross_entropy_loss::<Tensor>(&policy, None, tch::Reduction::Mean, -100, 0.0);
            let mse = evaluated_reward.mse_loss(&reward, tch::Reduction::Mean);
            let loss = &cross_entropy + &mse;
            
            // Compute the (temporary) gradients
            self.optimizers[i].zero_grad();
            loss.backward();

            // bring back the loss values to CPU
            let cross_entropy =  cross_entropy.to(Device::Cpu); 
            let mse = mse.to(Device::Cpu);

            //  add to the total losses
            cross_entropy_loss += cross_entropy.double_value(&[]);
            mse_loss += mse.double_value(&[]);
            total_loss += loss.double_value(&[]);
            
            self.step_count += 1;
        }

        let denominator: f64 = self.replicas().len() as f64;
        return LossValue::loss_value( total_loss / denominator, cross_entropy_loss / denominator, mse_loss / denominator)
    }

    fn sync_grads(&mut self) -> Result<()>{

        if self.replicas().len() == 1 {
            return Ok(());
        }
        
        let n_model = self.replicas().len();
        if n_model <= 1 {
            return Err(ErrReport::msg("no need to sync"));
        }
        let device = Device::Cpu;

        for name in self.trainable_var_names.clone().iter() {

            let mut vs_iter = self.replicas().iter();
            let mut sum = vs_iter.next().unwrap()
                .1
                .variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .grad()
                .copy()
                .to(device);

            for vs in vs_iter{
                sum += vs
                    .1
                    .variables()
                    .get(name)
                    .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                    .grad()
                    .copy()
                    .to(device);
            }
            
            let denom = tch::Scalar::float(n_model as f64);
            let reduced_grad = match self.sync_config.grad {
                SumOrAve::Ave => {
                    sum.divide_scalar(denom)
                }
                SumOrAve::Sum => {
                    sum
                }
            };

            let replicas_len = self.replicas().len();
            let ptr = self as *mut Self;
            // SAFETY: We ensure exclusive access for mutation here.
            for i in 0..replicas_len {
                let vs = unsafe { &mut (*ptr).shelf.get_group_mut(0).unwrap()[i].1 };
                let value = reduced_grad.to_device(vs.device());
                vs.variables()
                    .get_mut(name)
                    .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                    .grad()
                    .copy_(&value);
            }
        }
        Ok(())
    }

    pub fn sync_bn_stats(&mut self) -> Result<()> {

        if self.replicas().len() == 1 {
            return Ok(());
        }
        let ptr = &raw const *self.bn_stats_names;
        let mut vss = self.shelf.get_group_mut(0).unwrap()
            .iter_mut()
            .map(|(_, vs)| vs)
            .collect::<Vec<_>>();

        sync_bn_stats(vss.as_mut_slice(), unsafe{ & (*ptr) })
    }

    pub fn step<D: BoardData, R>(&mut self, replay_buffer: &ReplayBuffer<D>, sync_bn_stats: bool, rng: &mut R) -> Result<()> 
    where
    ReplayBuffer<D>: Sampler,
    R: Rng{

        if self.replicas().len() == 1 {

            let ptr = &raw mut self.optimizers;
            let batch = replay_buffer.sample_batch(self.mini_batch_size, (Kind::Float, Device::Cpu), rng)?;
            let loss_value = train_from_batch(
                &self.replicas().get(0).ok_or_eyre("Could not get model replica")?.0,
                unsafe {&mut *ptr}.get_mut(0).ok_or_eyre("Could not mutably get optimizer")?,
                batch
            );
            self.step_count += 1;

            #[cfg(feature = "verbose_lvl2")]
            println!("{loss_value}");

            self.loss_record.push(loss_value);

        } else {

            let loss_value = self.forward_backward_pass(replay_buffer, rng);
            self.loss_record.push(loss_value);
            self.sync_grads()?;
            if sync_bn_stats {
                self.sync_bn_stats()?;
            }
            for optimizer in self.optimizers.iter_mut() {
                optimizer.step();
            }
        }
        Ok(())
    }

    pub fn train<D: BoardData, B: Borrow<ReplayBuffer<D>>, R>(&mut self, n_steps: usize, sync_bn_stats_every: usize, replay_buffer: B, rng: &mut R) -> Result<()> 
    where
    ReplayBuffer<D>: Sampler,
    R: Rng
    {

        for i in 0..n_steps {
            let sync_bn_stats: bool = 
                if i > 0 && i % sync_bn_stats_every == 0 { true } 
                else if i == n_steps - 1 {true} 
                else { false }
            ; 
            self.step(replay_buffer.borrow(), sync_bn_stats, rng)?;

            #[cfg(feature = "verbose_lvl1")]
            println!("Train: {}/{}", i, n_steps);
        }
        Ok(())
    }

    pub fn save_to_stream<W: Write>(&self, stream: W) -> Result<()> {
        self.replicas()[0].1.save_to_stream(stream).wrap_err("saving model parameter failed")
    }

    pub fn loss_record(&self) -> &[LossValue] {
        &self.loss_record
    }

    pub fn loss_record_mut(&mut self) -> &mut[LossValue] {
        &mut self.loss_record
    }

    pub fn write_loss_record<W: Write>(&self, f: &mut W) {
        for loss in self.loss_record() {
            writeln!(f, "{:.4} {:.4} {:.4}", loss.cross_entropy, loss.mse, loss.total);
        }
    }

    pub fn flush_loss_record<W: Write>(&mut self, f: &mut W) {
        let drain = self.loss_record.drain(..);
        for loss in drain {
            writeln!(f, "{:.4} {:.4} {:.4}", loss.cross_entropy, loss.mse, loss.total);
        }
    }

    pub fn update_lr(&mut self, new_lr: f64) {
        for o in self.optimizers.iter_mut() {
            o.set_lr(new_lr);
        }
    }
}


pub struct Trainer<'a, P: PVModel> {
    replicas: &'a mut Vec<(P, VarStore)>,
    optimizers: Vec<Optimizer>,
    mini_batch_size: usize,
    trainable_var_names: Vec<String>,
    bn_stats_names: Vec<String>,
    sync_config: ModelSyncConfig,
    loss_record: Vec<LossValue>,
    rng: ThreadRng,
}

impl<'a, P: PVModel> Trainer<'a, P> {
    pub fn new<O: OptimizerConfig + Clone>(
        replicas: &'a mut Vec<(P, VarStore)>,
        config: O,
        lr: f64,
        weight_decay: f64,
        mini_batch_size: usize,
        sync_config: ModelSyncConfig
    ) -> Result<Self>{

        if replicas.len() == 0 {
            return Err(eyre!("models has to have at least one element"));
        }

        let optimizers= replicas.iter().map(|(module, vs)| {

            if let Ok(mut optim) = config.clone().build(vs, lr) {
                optim.set_weight_decay_group(NO_WEIGHT_DECAY_GROUP, weight_decay);
                Ok(optim)
            } else {
                Err(eyre!("Failed to build OptimConfig"))
            }
        }).collect::<Result<Vec<_>>>()?;

        let vss = replicas.iter().map(|(_, vs)| {
            vs
        }).collect::<Vec<_>>();
        // Get variable names in the nn weight for whom to sync the grads
        let trainable_var_names = get_trainable_var_names(&vss)?;

        // Get batch norm layer stats names
        let mut bn_stats_names = get_bn_mean_names(&vss).ok_or_eyre("Could not get batch norm mean names")?;
        bn_stats_names.extend(get_bn_var_names(&vss).ok_or_eyre("Could not get batch norm var names")?);

        let loss_record: Vec<LossValue> = vec![];
        let rng = thread_rng();
        Ok(Self { replicas, optimizers, mini_batch_size, trainable_var_names, bn_stats_names, sync_config, loss_record, rng})
    }

    // Sample from the replay_buffer, calculate the forward pass on each replica and return the average loss value
    fn forward_backward_pass<D: BoardData>(&mut self, replay_buffer: &ReplayBuffer<D>) -> LossValue 
    where ReplayBuffer<D>: Sampler {

        let mut cross_entropy_loss = 0.0f64;
        let mut mse_loss = 0.0f64;
        let mut total_loss = 0.0f64;

        for ((replica, _), optimizer) in self.replicas.iter().zip(self.optimizers.iter_mut()) {

            // sample mini-batch from the replay buffer and do the forward pass
            let device = replica.device();
            let (mut position, mut policy, mut reward) 
                = replay_buffer.sample_batch(self.mini_batch_size, (Kind::Float, Device::Cpu), &mut self.rng).unwrap();
            position = position.to_device(device);
            policy = policy.to_device(device);
            reward = reward.to_device(device);
            let (evaluated_logits, evaluated_reward) = replica.evaluate_t(&position, true);
            let cross_entropy = evaluated_logits.cross_entropy_loss::<Tensor>(&policy, None, tch::Reduction::Mean, -100, 0.0);
            let mse = evaluated_reward.mse_loss(&reward, tch::Reduction::Mean);
            let loss = &cross_entropy + &mse;
            
            // Compute the (temporary) gradients
            optimizer.zero_grad();
            loss.backward();

            // bring back the loss values to CPU
            let cross_entropy =  cross_entropy.to(Device::Cpu); 
            let mse = mse.to(Device::Cpu);

            //  add to the total losses
            cross_entropy_loss += cross_entropy.double_value(&[]);
            mse_loss += mse.double_value(&[]);
            total_loss += loss.double_value(&[]);
            
        }

        let denominator: f64 = self.replicas.len() as f64;
        return LossValue::loss_value( total_loss / denominator, cross_entropy_loss / denominator, mse_loss / denominator)
    }

    fn sync_grads(&mut self) -> Result<()>{

        if self.replicas.len() == 1 {
            return Ok(());
        }
        let mut vss = self.replicas.iter_mut().map(|(_, vs)| { vs }).collect::<Vec<_>>();
        sync_grads(vss.as_mut_slice(), self.trainable_var_names.as_slice(), self.sync_config)
    }

    fn sync_bn_stats(&mut self) -> Result<()> {

        if self.replicas.len() == 1 {
            return Ok(());
        }
        let mut vss = self.replicas.iter_mut().map(|(_, vs)| { vs }).collect::<Vec<_>>();
        sync_bn_stats(vss.as_mut_slice(), self.bn_stats_names.as_slice())
    }

    fn step<D: BoardData>(&mut self, replay_buffer: &ReplayBuffer<D>, sync_bn_stats: bool) -> Result<()> 
    where ReplayBuffer<D>: Sampler {

        if self.replicas.len() == 1 {

            let batch = replay_buffer.sample_batch(self.mini_batch_size, (Kind::Float, Device::Cpu), &mut self.rng)?;
            let loss_value = train_from_batch(
                &self.replicas.get(0).ok_or_eyre("Could not get model replica")?.0,
                self.optimizers.get_mut(0).ok_or_eyre("Could not mutably get optimizer")?,
                batch
            );

            if cfg!(test) {
                print!("{loss_value}");
            }
            self.loss_record.push(loss_value);

        } else {

            let loss_value = self.forward_backward_pass(replay_buffer);
            self.loss_record.push(loss_value);
            self.sync_grads()?;
            if sync_bn_stats {
                self.sync_bn_stats()?;
            }
            for optimizer in self.optimizers.iter_mut() {
                optimizer.step();
            }
        }
        Ok(())
    }

    pub fn train<D: BoardData, B: Borrow<ReplayBuffer<D>>>(&mut self, n_steps: usize, sync_bn_stats_every: usize, replay_buffer: B) -> Result<()> 
    where ReplayBuffer<D>: Sampler {

        for i in 0..n_steps {
            let sync_bn_stats: bool = 
                if i > 0 && i % sync_bn_stats_every == 0 { true } 
                else if i == n_steps - 1 {true} 
                else { false }
            ; 
            self.step(replay_buffer.borrow(), sync_bn_stats)?;

            #[cfg(feature = "verbose_lvl1")]
            println!("Train {}/{}", i, n_steps);

        }
        Ok(())
    }

    pub fn save_to_stream<W: Write>(&self, stream: W) -> Result<()> {
        self.replicas[0].1.save_to_stream(stream).wrap_err("saving model parameter failed")
    }
}

fn train_from_batch<P: PVModel>(model: &P, optimizer: &mut Optimizer, batch: (Tensor, Tensor, Tensor)) -> LossValue {

    let device = model.device();
    let (mut position, mut policy, mut reward) = batch;
    position = position.to_device(device);
    policy = policy.to_device(device);
    reward = reward.to_device(device);
    let (evaluated_logits, evaluated_reward) = model.evaluate_t(&position, true);
    // dbg!(evaluated_logits.size());
    // dbg!(policy.size());
    let cross_entropy = evaluated_logits.cross_entropy_loss::<Tensor>(&policy, None, tch::Reduction::Mean, -100, 0.0);
    let mse = evaluated_reward.mse_loss(&reward, tch::Reduction::Mean);
    let loss = &cross_entropy + &mse;

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    let total = loss.to(Device::Cpu).double_value(&[]);
    let mse = mse.to(Device::Cpu).double_value(&[]);
    let cross_entropy = cross_entropy.to(Device::Cpu).double_value(&[]);
    LossValue::loss_value(total, cross_entropy, mse)
}
