//TODO!
// design a nn with policy and value head

#![cfg(feature = "torch")]
#![allow(unused)]
#![allow(dead_code)]
//external
use tch;
use tch::nn;
use serde::Deserialize;
use color_eyre::eyre::ErrReport;

//internal
use game::board::TaflBoardEleven;
use bitboard::eleven::MoveOnBoardEleven;
use tch::nn::ConvConfigND;
use tch::nn::Module;

// std
use std::fs;

#[derive(Debug, Deserialize)]
struct ModelConfig {
    learning_rate: f32,
    batch_size: i64,
    input_history_length: usize,
    policy_out_features: i64,
    board_size: i64,
    value_hidden_layer_unit_count: i64,

    // Conv block related
    num_convblocks: usize,
    in_features: i64,
    conv_filters: i64,
    kernel_size: i64,
}

fn load_model_config(path: &str) 
-> Result<ModelConfig, ErrReport> {
    let config_str = fs::read_to_string(path)?;
    let params: ModelConfig = toml::from_str(&config_str)?;
    Ok(params)
}

fn conv_block(
    vs: &nn::Path,
    in_features: i64,
    out_features: i64, 
    ksize: i64,
    id: usize) -> nn::Conv2D {

    nn::conv2d(vs / &format!("conv block {}", id), in_features, out_features, ksize, Default::default())

}
// pub struct ConvBlock {
//     conv: nn::Conv2D,
// }

// impl ConvBlock {
//     fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
//         let i = config.in_features;
//         let o = config.conv_filters;
//         let k = config.kernel_size;
//         let conv1 = nn::conv2d(vs,i,o,k,Default::default());
//     }
// }

// impl nn::ModuleT for ConvBlock {
//     fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
//         xs.apply(&self.conv)
//             .relu()
//     }
// } 

fn base_net(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {

    let mut seq = nn::seq();
    for i in 0..config.num_convblocks {

        let in_features = if i == 0 { config.in_features } else { config.conv_filters };
        let out_features = config.conv_filters;
        let ksize = config.kernel_size;
        let id = i + 1;
        seq = seq.add(conv_block(vs, in_features, out_features, ksize, id));
        seq = seq.add_fn(move |ts| ts.zero_pad2d((ksize - 1) / 2, (ksize - 1) / 2, (ksize - 1) / 2, (ksize - 1) / 2));
        seq = seq.add_fn(|ts| ts.relu());

    }

    seq

}

fn policy_head(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
    
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

fn value_head(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
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

struct PVNet {
    base_net: nn::Sequential,
    policy_head: nn::Sequential,
    value_head: nn::Sequential,
}

impl PVNet {
    pub fn model(vs: &nn::Path, config: &ModelConfig) -> Self {
        let base_net = base_net(&(vs / "base"), config);
        let policy_head = policy_head(&(vs / "policy_head"), config);
        let value_head = value_head(&(vs / "value_head"), config);

        Self { base_net, policy_head, value_head } 
    }
}

impl PVNet {
    pub fn infer(&self, xs: &tch::Tensor) -> (tch::Tensor, tch::Tensor) {
        let xs = self.base_net.forward(xs);
        let p = self.policy_head.forward(&xs);
        let v = self.value_head.forward(&xs);
        (p, v)
    }
}

fn pnet(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    seq = seq.add(base_net(&(vs / "base"), config));
    seq = seq.add(policy_head(&(vs / "head"), config));
    seq
}

fn vnet(vs: &nn::Path, config: &ModelConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    seq = seq.add(base_net(&(vs / "base"), config));
    seq = seq.add(value_head(&(vs / "head"), config));
    seq
}

#[cfg(test)]
mod tests {
    
    use crate::model::*;
    const CONFIG: ModelConfig = ModelConfig {
        learning_rate: 0.01,
        batch_size: 10,
        input_history_length: 4,
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