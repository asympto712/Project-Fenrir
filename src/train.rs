#![allow(dead_code)]
#![allow(unused)]

use std::collections::HashMap;

use crate::model::*;
use crate::model::PVModel::*;
use crate::replay_buffer::ReplayBuffer;

#[cfg(feature = "torch")]
use tch::nn::{self, VarStore, Variables, Path};
#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};
#[cfg(feature = "torch")]
use tch::nn::{Optimizer, OptimizerConfig};

use color_eyre::eyre::{eyre, ErrReport, Result};


#[derive(Debug, Clone, Copy, PartialEq)]
enum SumOrAve {
    Sum,
    Ave,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ModelSyncConfig{
    grad: SumOrAve
}

pub struct LossValue {
    pub total: f32,
    pub cross_entropy: f32,
    pub mse: f32,
}

impl LossValue {
    fn loss_value(total: f32, cross_entropy: f32, mse: f32) -> Self {
        Self { total, cross_entropy, mse }
    }
}

fn sync_grads<T: AsRef<[VarStore]>>(vss: T, names: Vec<String>, config: ModelSyncConfig) -> Result<()> {

    let n_model = vss.as_ref().len();
    if n_model <= 1 {
        return ErrReport::msg("no need to sync");
    }
    let n_model = tch::Scalar::float(n_model as f64);
    let device = Device::Cpu;

    for name in names.iter() {

        let mut vs_iter = vss.as_ref().iter();
        let mut sum = vs_iter.next().unwrap()
            .variables()
            .get(name)
            .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
            .grad()
            .copy()
            .to(device);

        for vs in vs_iter{
            sum += vs
                .variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .grad()
                .copy()
                .to(device);
        }
        
        let reduced_grad = match config.grad {
            SumOrAve::Ave => {
                sum.divide_scalar(n_model)
            }
            SumOrAve::Sum => {
                sum
            }
        };

        for vs in vss.as_ref.iter(){
            let mut grad = vs.variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .grad();

            reduced_grad.to_device(grad.device());
            grad.copy_(&reduced_grad);
        }
    }
}

// The paper from facebook (https://arxiv.org/abs/1706.02677) claims that the syncing of 
// mean and variance estimate per single mini-batch is redundant and results in the inconsistent loss functions across the workers.
// This point is quite the contrary to the official DDP implementation by pytorch (SyncBatchNorm),
// and chatGPT seems to strongly side with the latter (I tried to 'convince' it multiple times).
// What seems to be the common consensus is that Batch-Norm can actually degrade the sample quality if used on exceptionally small (2 ~ 8) mini-batches.
// syncing the statistics per mini-batch across workers is not ergonomic with the tch-rs framework,
// at least not while using the standard API (one probably has to create custom SyncBatchNorm struct for that from scratch).
// In conclusion, I have currently no other way than only sync the running stats across workers at the end of one epoch, not per mini-batch.
// Whether 32 samples per worker (as used by AlphaZero) is sufficient to justify this simplification is debatable,
// and I might have to observe the actual statistics of each neuron activation.
// On the other hand, I might be better off using other normalization technique than BN, such as Group N or Layer N.
// Using Layer Normalization will remove the need for this additional sync completely.
fn sync_bn_stats<T: AsRef<[VarStore]>>(vss: T, names: Vec<String>) -> Result<()> {

    let n_model = vss.as_ref.len();
    if n_model <= 1 {
        return ErrReport::msg("no need to sync");
    }
    let n_model = tch::Scalar::float(n_model as f64);
    let device = Device::Cpu;

    for name in names.iter() {

        let mut vs_iter = vss.as_ref.iter();
        let mut sum = vs_iter.next().unwrap()
            .variables()
            .get(name)
            .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
            .copy()
            .to(device);

        for vs in vs_iter{
            sum += vs
                .variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?
                .copy()
                .to(device);
        }

        let mut ave = sum.divide_scalar(n_model);

        for vs in vss.as_ref.iter(){
            let mut stat = vs.variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?;

            ave.to_device(stat.device());
            stat.copy_(&ave);
        }
    }
    Ok(())

}

fn get_trainable_var_names<T: AsRef<[VarStore]>>(vss: T) -> Result<Vec<String>> {

    let var_store = vss.as_ref()[0];
    let len = var_store.len();
    if !vss.as_ref().iter().all(|vs| vs.len() == len) {
        return Err(eyre!("input VarStores don't share the same length"));
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("weight") || key.contains("bias") {

            for vs in vss.as_ref.iter(){
                let ts = vs
                    .variables()
                    .get(key.as_str())
                    .ok_or_else(eyre!("variable {} does not exist in one of the VarStores", key))?;

                if ts.requires_grad() {
                    names.push(key.clone());
                } else {
                    return Err(eyre!("Variable {} does not track its gradient!", key));
                }
            }

        }
    }

    Ok(names)
}

fn get_bn_mean_names<T: AsRef<[VarStore]>>(vss: T) -> Option<Vec<String>> {

    let var_store = vss.as_ref()[0];
    let len = var_store.len();
    if !vss.iter().all(|vs| vs.len() == len) {
        return None;
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("running_mean") {

            for vs in vss.as_ref().iter(){
                let opt = vs.variables().get(key.as_str());
                match opt {
                    Some(_) => {
                        names.push(key.clone());
                    },
                    None => {
                        return None;
                    }
                }
            }

        }
    }

    Some(names)

}

fn get_bn_var_names<T: AsRef<[VarStore]>>(vss: T) -> Option<Vec<String>> {

    let var_store = vss.as_ref()[0];
    let len = var_store.len();
    if !vss.iter().all(|vs| vs.len() == len) {
        return None;
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("running_var") {

            for vs in vss.as_ref().iter(){
                let opt = vs.variables().get(key.as_str());
                match opt {
                    Some(_) => {
                        names.push(key.clone());
                    },
                    None => {
                        return None;
                    }
                }
            }

        }
    }

    Some(names)


}

fn train_from_batch<P: PVModel>(model: &P, optimizer: &mut Optimizer, batch: (Tensor, Tensor, Tensor)) -> LossValue {

    let device = model.device();
    let (mut position, mut policy, mut reward) = batch;
    position = position.to_device(device);
    policy = policy.to_device(device);
    reward = reward.to_device(device);
    let (evaluated_logits, evaluated_reward) = model.evaluate_t(&position, true);
    let cross_entropy = evaluated_logits.cross_entropy_for_logits(&policy);
    let mse = evaluated_reward.mse_loss(&reward, tch::Reduction::Mean);
    let loss = &cross_entropy + &mse;

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    let total = loss.to(Device::Cpu).double_value(&[0]) as f32;
    let mse = loss.to(Device::Cpu).double_value(&[0]) as f32;
    let cross_entropy = loss.to(Device::Cpu).double_value(&[0]) as f32;
    LossValue::loss_value(total, cross_entropy, mse)
}
