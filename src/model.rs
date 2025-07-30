//TODO!
// design a nn with policy and value head

#![cfg(feature = "torch")]
#![allow(unused)]
#![allow(dead_code)]
use bincode::config::Config;
//external
use tch;
use tch::nn;
use tch::nn::linear;
use tch::nn::VarStore;
use tch::Device;
use tch::Kind;
use tch::Tensor;
use tch::nn::ConvConfigND;
use tch::nn::Module;
use tch::nn::ModuleT;
use tch::nn::FuncT;
use serde::Deserialize;
use color_eyre::eyre::ErrReport;
use color_eyre::eyre::Result;


use std::collections::HashMap;
use std::fmt::format;
// std
use std::fs;

// It seems like theoretically, applying weight decay such as L2 regularization does not make much sense,
// although the degradation effect (or lack thereof) on the actual performance by doing so is debatable.
// For reference, official pytorch does include BN weight and bias in the weight decay group.
// If one decide to exclude them, one can simply call .set_group method on the Path instance to pass to the batch-norm creation function
// and adjust the weight decay group in the optimizer accordingly.
pub const NO_WEIGHT_DECAY_GROUP: usize = 1;

pub const MOCK_MODEL_CONFIG: ModelConfig = ModelConfig{
    policy_out_features: 0,
    board_size: 0,
    value_hidden_layer_unit_count: 0,
    num_convblocks: 0,
    in_features: 0,
    conv_filters: 0,
    kernel_size: 0,
};

pub trait PVModel {
    fn new(vs: &nn::Path, config: &ModelConfig) -> Self; 
    fn evaluate_t(&self, xs: &Tensor, train: bool) -> Evaluation;
    fn device(&self) -> Device;
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    policy_out_features: i64,
    board_size: i64,
    value_hidden_layer_unit_count: i64,

    // Conv block related
    num_convblocks: usize,
    in_features: i64,
    conv_filters: i64,
    kernel_size: i64,
}

impl ModelConfig {
    pub fn load_from_toml(path: &str) -> Result<Self>{
        let config_str = fs::read_to_string(path)?;
        let params: ModelConfig = toml::from_str(&config_str)?;
        Ok(params)
    }
    pub fn new(
        policy_out_features: i64,
        board_size: i64,
        value_hidden_layer_unit_count: i64,
        num_convblocks: usize,
        in_features: i64,
        conv_filters: i64,
        kernel_size: i64,
    ) -> Self {
        Self { policy_out_features, board_size, value_hidden_layer_unit_count, num_convblocks, in_features, conv_filters, kernel_size}
    }
}

fn load_model_config(path: &str) 
-> Result<ModelConfig, ErrReport> {
    let config_str = fs::read_to_string(path)?;
    let params: ModelConfig = toml::from_str(&config_str)?;
    Ok(params)
}


// Expects to use odd kernel size. Keep the 2d dimension unchanged by padding
fn conv_config(ksize: i64) -> nn::ConvConfig {
    let pad: i64 = (ksize - 1) / 2;
    nn::ConvConfig {
        stride: 1,
        padding: pad,
        ..Default::default()
    }
}

// in default, cudNN is enabled, affine learnable transformation is on, and initial value is based on that BN paper (2016)
fn bn_config() -> nn::BatchNormConfig {
    nn::BatchNormConfig {
        ..Default::default()
    }
}

// function that defines residual block specified in AlphaZero. Kernel size is 3 across the board
// Assumes that in_feature = out_feature
fn res_block(
    vs: &nn::Path,
    features: i64,
    ksize: i64,
) -> impl ModuleT + use<> {
    
    // setting the weight and bias of batch norm as exception for weight decay
    let vs_no_weight_decay = vs.set_group(NO_WEIGHT_DECAY_GROUP);
    let conv1 = nn::conv2d(vs / "conv1", features, features, ksize, conv_config(ksize));
    let bn1 = nn::batch_norm2d(&(vs_no_weight_decay.clone() / "bn1"), features, bn_config());
    let conv2 = nn::conv2d(vs / "conv2", features,features, ksize, conv_config(ksize));
    let bn2 = nn::batch_norm2d(&(vs_no_weight_decay / "bn2"), features, bn_config());
    nn::func_t(move |xs, train| {
        let ys = xs.apply(&conv1).apply_t(&bn1, train).relu().apply(&conv2).apply_t(&bn2, train);
        (xs + ys).relu()
    })

}

fn conv_block(
    vs: &nn::Path,
    in_features: i64,
    out_features: i64,
    ksize: i64,
) -> impl ModuleT + use<> {

    let vs_no_weight_decay = vs.set_group(NO_WEIGHT_DECAY_GROUP);
    let conv = nn::conv2d(vs / "conv1", in_features, out_features, ksize, conv_config(ksize));
    let bn = nn::batch_norm2d(vs_no_weight_decay / "bn1", out_features, bn_config());
    nn::func_t(move |xs, train| {
        xs.apply(&conv).apply_t(&bn, train).relu()
    })

}

fn base_tower(
    vs: &nn::Path,
    config: &ModelConfig
) -> nn::SequentialT {

    let mut seq = nn::seq_t();
    seq = seq.add(conv_block(&(vs / "block0"), config.in_features, config.conv_filters, config.kernel_size));
    for i in 1..=config.num_convblocks {

        seq = seq.add(res_block(&(vs / &format!("block{}", i)), config.conv_filters, config.kernel_size));
    }
    seq
}

fn fenrir_policy_head(
    vs: &nn::Path,
    config: &ModelConfig
) -> nn::SequentialT {

    nn::seq_t()
        .add(conv_block(&(vs / "conv layer"), config.conv_filters, config.conv_filters, config.kernel_size))
        .add(conv_block(&(vs / "logit layer"), config.conv_filters, config.policy_out_features, 1))
}

fn fenrir_value_head(
    vs: &nn::Path,
    config: &ModelConfig
) -> nn::SequentialT {

    nn::seq_t()
        .add(nn::conv2d(vs / "conv layer", config.conv_filters, 1, 1, Default::default())) // (BS, 1, N, N)
        .add_fn(|ts| ts.view([ts.size()[0], -1])) // (BS, N * N)
        .add(linear(vs / "linear", config.board_size * config.board_size, 256, Default::default()))
        .add_fn(|ts| ts.relu())
        .add(linear(vs / "linear2", 256, 1, Default::default()))
        .add_fn(|ts| ts.tanh())
}

struct GeneralPVDualModel {
    base: nn::SequentialT,
    p_head: nn::SequentialT,
    v_head: nn::SequentialT,
    config: ModelConfig,
    device: Device,
}

impl PVModel for GeneralPVDualModel {
    fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        
        let base = base_tower(&(vs / "base"), &config);
        let p_head = fenrir_policy_head(&(vs / "p-head"), &config);
        let v_head = fenrir_value_head(&(vs / "v-head"), &config);
        Self { base, p_head, v_head, config: config.clone(), device: vs.device()}
    }
    // Assumes that the input has the shape (BS, Features, N, N), with the last feature being the playing side
    fn evaluate_t(&self, xs: &Tensor, train: bool) -> Evaluation {
        
        let v = xs.split_sizes([-1,1], 1);
        let minus_side = v[0].contiguous();
        let based = minus_side.apply_t(&self.base, train);
        // (BS, Conv_filters, N, N)
        let base_output = Tensor::cat(&[&based, &v[1]], 1);
        let base_output = base_output.contiguous(); // Again, not sure if needed
        let logits = base_output.apply_t(&self.p_head, train);
        let value = base_output.apply_t(&self.v_head, train);

        if !train {
            let likelihood = logits
                .view([logits.size1().unwrap(), -1])
                .softmax(1, Kind::Float)
                .view([logits.size1().unwrap(), self.config.policy_out_features, self.config.board_size, self.config.board_size]);
            (likelihood, value)
        } else {
            (logits, value)
        }
    }

    fn device(&self) -> Device {
        self.device
    }
}


struct GeneralPVSepModel {
    p_base: nn::SequentialT,
    v_base: nn::SequentialT,
    p_head: nn::SequentialT,
    v_head: nn::SequentialT,
    config: ModelConfig,
    device: Device,
}

impl PVModel for GeneralPVSepModel {
    fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        
        let p_base = base_tower(&( vs / "p-base" ), &config);
        let v_base = base_tower(&( vs / "v-base" ), &config);
        let p_head = fenrir_policy_head(&( vs / "p-head" ), &config);
        let v_head = fenrir_value_head(&( vs / "v-head" ), &config);
        Self { p_base, v_base, p_head, v_head, config: config.clone(), device: vs.device()}
    }
    // Assumes that the input has the shape (BS, Features, N, N), with the last feature being the playing side
    fn evaluate_t(&self, xs: &Tensor, train: bool) -> Evaluation {

        let v = xs.split_sizes([-1, 1], 1);
        let minus_side = v[0].contiguous(); // Might not need this since conv2D automatically creates new contiguous tensor (needs testing)
        let logits = xs.apply_t(&self.p_base, train).apply_t(&self.p_head, train);
        let value = minus_side.apply_t(&self.v_base, train).apply_t(&self.v_head, train);
        if !train {
            let likelihood = logits
                .view([logits.size1().unwrap(), -1])
                .softmax(1, Kind::Float)
                .view([logits.size1().unwrap(), self.config.policy_out_features, self.config.board_size, self.config.board_size]);
            (likelihood, value)
        } else {
            (logits, value)
        }
    }

    fn device(&self) -> Device {
        self.device
    }

}



fn alpha_go_conv_block(
    vs: &nn::Path,
    in_features: i64,
    out_features: i64, 
    ksize: i64,
    id: usize) -> nn::Conv2D {

    nn::conv2d(vs / &format!("conv block {}", id), in_features, out_features, ksize, conv_config(ksize))
}

fn alpha_go_base_net(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {

    let mut seq = nn::seq();
    for i in 0..config.num_convblocks {

        let in_features = if i == 0 { config.in_features } else { config.conv_filters };
        let out_features = config.conv_filters;
        let ksize = config.kernel_size;
        let id = i + 1;
        seq = seq.add(alpha_go_conv_block(vs, in_features, out_features, ksize, id));
        seq = seq.add_fn(|ts| ts.relu());

    }

    seq

}

fn alpha_go_policy_head(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
    
    let mut seq = nn::seq();
    let base_out_features = config.conv_filters;
    let board_size = config.board_size;
    let policy_out_features = config.policy_out_features;
    seq = seq.add(nn::conv2d(vs / "final conv layer", base_out_features,policy_out_features, 1, Default::default()));
    // (batch_size, policy_out_features, board_size, board_size)
    seq = seq.add_fn(move |ts| ts.view_([-1, board_size * board_size * policy_out_features]));
    // (batch_size, -1)
    seq = seq.add_fn(|ts| ts.softmax(1, tch::Kind::Float));
    seq = seq.add_fn(move |ts| ts.view([-1, policy_out_features, board_size, board_size]));
    seq
}

fn alpha_go_value_head(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    let base_out_features = config.conv_filters;
    let board_size = config.board_size;
    let value_hidden_layer_unit_count = config.value_hidden_layer_unit_count;
    seq = seq.add(nn::conv2d(vs / "final conv layer", base_out_features, 1, 1, Default::default()));
    seq = seq.add_fn(|ts| ts.reshape([ts.size()[0], -1]));
    seq = seq.add(nn::linear(vs / "hidden linear", board_size * board_size, value_hidden_layer_unit_count, Default::default()));
    seq = seq.add_fn(|ts| ts.relu());
    seq = seq.add(nn::linear(vs / "final layer", value_hidden_layer_unit_count, 1, Default::default()));
    seq = seq.add_fn(|ts| ts.tanh());
    seq
}

pub type Evaluation = (Tensor, Tensor);

#[derive(Debug)]
pub struct PVNet {
    base_net: nn::Sequential,
    policy_head: nn::Sequential,
    value_head: nn::Sequential,
}

impl PVNet {
    pub fn model(vs: &nn::Path, config: &ModelConfig) -> Self {
        let base_net = alpha_go_base_net(&(vs / "base"), config);
        let policy_head = alpha_go_policy_head(&(vs / "policy_head"), config);
        let value_head = alpha_go_value_head(&(vs / "value_head"), config);

        Self { base_net, policy_head, value_head } 
    }
}

impl PVNet {
    pub fn infer(&self, xs: &tch::Tensor) -> Evaluation {
        let xs = self.base_net.forward(xs);
        let p = self.policy_head.forward(&xs);
        let v = self.value_head.forward(&xs);
        (p, v)
    }
}

fn pnet(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    seq = seq.add(alpha_go_base_net(&(vs / "base"), config));
    seq = seq.add(alpha_go_policy_head(&(vs / "head"), config));
    seq
}

fn vnet(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    seq = seq.add(alpha_go_base_net(&(vs / "base"), config));
    seq = seq.add(alpha_go_value_head(&(vs / "head"), config));
    seq
}

#[cfg(test)]
mod tests {
    
    use crate::model::*;
    const CONFIG: ModelConfig = ModelConfig {
        policy_out_features: 3,
        board_size: 5,
        value_hidden_layer_unit_count: 3, 
        num_convblocks: 1,
        in_features: 3, 
        conv_filters: 3,
        kernel_size: 3
    };
    use tch::{Kind, Device};
    use tch::Tensor;
    use tch::nn;

    #[test]
    fn vnet_gen_works() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = CONFIG;
        let net = vnet(&vs.root(), &config);
        for (name, v) in vs.variables() {
            println!("{}: {}", name, v);
        }
        let xs = tch::Tensor::rand([1,config.in_features, config.board_size, config.board_size], (tch::Kind::Float, tch::Device::Cpu));
        let output = net.forward(&xs);
        println!("{}", output);
        assert_eq!(output.size(), [1,1]);
        assert!(output.is_contiguous());
    }

    #[test]
    fn pnet_gen_works() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = CONFIG;
        let net = pnet(&vs.root(), &config);
        let xs = tch::Tensor::rand([1,config.in_features, config.board_size, config.board_size], (Kind::Float, Device::Cpu));
        let output = net.forward(&xs);
        assert_eq!(output.size(), [1,config.policy_out_features, config.board_size, config.board_size]);
        assert!(output.is_contiguous());
    }

}