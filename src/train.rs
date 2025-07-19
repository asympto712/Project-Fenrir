#![allow(dead_code)]
#![allow(unused)]

use std::collections::HashMap;

use crate::model::*;
use crate::replay_buffer::ReplayBuffer;

#[cfg(feature = "torch")]
use tch::nn::{self, VarStore, Variables, Path};
#[cfg(feature = "torch")]
use tch::Tensor;
#[cfg(feature = "torch")]
use tch::{Kind, Device};

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

fn sync_grads(vss: Vec<VarStore>, names: Vec<String>, config: ModelSyncConfig) -> Result<()> {

    let n_model = vss.len();
    if n_model <= 1 {
        return ErrReport::msg("no need to sync");
    }
    let n_model = tch::Scalar::float(n_model as f64);
    let device = Device::Cpu;

    for name in names.iter() {

        let mut vs_iter = vss.iter();
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

        for vs in vss.iter(){
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
fn sync_bn_stats(vss: Vec<VarStore>, names: Vec<String>) {

    let n_model = vss.len();
    if n_model <= 1 {
        return ErrReport::msg("no need to sync");
    }
    let n_model = tch::Scalar::float(n_model as f64);
    let device = Device::Cpu;

    for name in names.iter() {

        let mut vs_iter = vss.iter();
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

        for vs in vss.iter(){
            let mut stat = vs.variables()
                .get(name)
                .ok_or_else(|| eyre!(format!("Couldn't find the variable {}", name)))?;

            ave.to_device(grad.device());
            stat.copy_(&ave);
        }
    }

}

fn get_trainable_var_names(vss: Vec<VarStore>) -> Result<Vec<String>> {

    let var_store = vss[0];
    let len = vs.len();
    if !vss.iter().all(|vs| vs.len() == len) {
        return Err(eyre!("input VarStores don't share the same length"));
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("weight") || key.contains("bias") {

            for vs in vss.iter(){
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

    Some(names)
}

fn get_bn_mean_names(vss: Vec<VarStore>) -> Option<Vec<String>> {

    let var_store = vss[0];
    let len = vs.len();
    if !vss.iter().all(|vs| vs.len() == len) {
        return None;
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("running_mean") {

            for vs in vss.iter(){
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

fn get_bn_var_names(vss: Vec<VarStore>) -> Option<Vec<String>> {

    let var_store = vss[0];
    let len = vs.len();
    if !vss.iter().all(|vs| vs.len() == len) {
        return None;
    }
    let variables = var_store.variables();
    let mut names: Vec<String> = vec![];

    for key in variables.keys(){
        // so far I've checked, these are the only name patterns used by tch for trainable variables
        if key.contains("running_var") {

            for vs in vss.iter(){
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

fn play_and_collect_data() {
    todo!()
}


