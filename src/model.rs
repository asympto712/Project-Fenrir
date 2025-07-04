//TODO!
// design a nn with policy and value head

//external
use tch;
use tch::nn;
use serde::Deserialize;
use color_eyre::eyre::{Err, ErrReport};

//internal
use game::game::TaflBoardEleven;
use game::game::MoveOnBoardEleven;
use tch::nn::conv;
use tch::nn::ConvConfigND;
use tch::nn::Module;

// std
use std::fs;

#[derive(Debug, Deserialize)]
struct ModelConfig {
    learning_rate: f32,
    batch_size: usize,
    input_history_length: usize,
    policy_out_features: usize,
    board_size: usize,
    value_hidden_layer_unit_count: usize,

    // Conv block related
    num_convblocks: u8,
    in_features: u16,
    conv_filters: u16,
    kernel_size: u16,
}

fn load_model_config(path: &str) 
-> Result<ModelConfig, ErrReport> {
    let config_str = fs::read_to_string(path)?;
    let params: ModelConfig = toml::from_str(&config_str)?;
    Ok(params)
}

fn conv_block(
    vs: &nn::Path,
    in_features: u16,
    out_features: u16, 
    ksize: u16,
    id: usize) -> nn::Conv2D {

    let vvs = vs.path(format!("conv block {}", id));
    nn::conv2d(vs / vvs, in_features, out_features, ksize, Default::default())

}
pub struct ConvBlock {
    conv: nn::Conv2D,
}

impl ConvBlock {
    fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let i = config.in_features;
        let o = config.conv_filters;
        let k = config.kernel_size;
        let conv1 = nn::conv2d(vs,i,o,k,ConvConfigND);
    }
}

impl nn::ModuleT for ConvBlock {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        xs.apply(&self.conv1)
            .relu()
    }
} 

fn base_net(vs: &nn::Path, config: &ModelConfig) -> impl nn::Module {

    let seq = nn::seq();
    for i in 0..config.num_convblocks {

        let in_features = if i == 0 { config.in_features } else { config.conv_filters };
        let out_features = config.conv_filters;
        let ksize = config.kernel_size;
        let id = i + 1;
        seq.add(conv_block(vs, in_features, out_features, ksize, id));
        seq.add_fn(|ts| ts.zero_pad2d((ksize - 1) / 2, (ksize - 1) / 2, (ksize - 1) / 2, (ksize - 1) / 2));
        seq.add_fn(|ts| ts.relu());

    }

}

fn policy_head(vs: &nn::Path, config: &ModelConfig) -> impl nn::Module {
    
    let seq = nn::seq();
    let base_out_features = config.conv_filters;
    seq.add(nn::conv2d(vs / "final conv layer", base_out_features,config.policy_out_features, 1, Default::default()));
    // (batch_size, policy_out_features, 11, 11)
    seq.add_fn(|ts| ts.view_([-1, config.board_size * config.board_size * config.policy_out_features]));
    seq.add_fn(|ts| ts.softmax(1, f16));
    seq.add_fn(|ts| ts.view_as([-1, config.board_size, config.board_size, config.policy_out_features]));
    seq
}

fn value_head(vs: &nn::Path, config: &ModelConfig) -> impl nn::Module{
    let seq = nn::seq();
    let base_out_features = config.conv_filters;
    seq.add(nn::conv2d(vs / "final conv layer", base_out_features, 1, 1, Default::default()));
    seq.add_fn(|ts| ts.reshape([ts.size()[0], -1]));
    seq.add(nn::linear(vs / "hidden linear", config.board_size * config.board_size, config.value_hidden_layer_unit_count, Default::default()));
    seq.add_fn(|ts| ts.relu());
    seq.add(nn.linear(vs / "final layer", config.value_hidden_layer_unit_count, 1, Default::default()));
    seq.add_fn(|ts| ts.tanh());
    seq
}

struct PVNet {
    base_net: nn::Module,
    policy_head: nn::Module,
    value_head: nn::Module,
}

impl PVNet {
    pub fn model(vs: &nn::Path, config: &ModelConfig) -> Self {
        let base_net = base_net(vs / "base", config);
        let policy_head = policy_head(vs / "policy_head", config);
        let value_head = value_head(vs / "value_head", config);

        Self { base_net, policy_head, value_head } 
    }
}

impl PVNet {
    pub fn infer(&self, xs: &tch::Tensor) -> (tch::Tensor, tch::Tensor) {
        let xs = self.base_net.forward(xs);
        let p = self.policy_head.forward(xs);
        let v = self.value_head.forward(xs);
        (p, v)
    }
}

fn pnet(vs: &nn::Path, config: &ModelConfig) -> impl nn::Module {
    let seq = nn::seq();
    seq.add(base_net(vs / "base", config));
    seq.add(policy_head(vs / "head", config));
    seq
}

fn vnet(vs: &nn::Path, config: &ModelConfig) -> impl nn::Module {
    let seq = nn::seq();
    seq.add(base_net(vs / "base", config));
    seq.add(value_head(vs / "head", config));
    seq
}

#[test]
fn pvnet_gen_works() {
    let config: ModelConfig = ModelConfig { 
        learning_rate: 0.01,
        batch_size: 10,
        input_history_length: 4,
        policy_out_features: 3,
        board_size: 5,
        value_hidden_layer_unit_count: 128, 
        num_convblocks: 3,
        in_features: 3, 
        conv_filters: 5,
        kernel_size: 3
    };

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let net = vnet(vs.root(), &config);
    let xs = tch::Tensor::rand([1,config.in_features, config.board_size, config.board_size], (tch::Kind::Float, tch::Device::Cpu));
    let result = net.forward(xs);
    for (name, v) in vs.variables() {
        println!("{}: {}", name, v);
    }
}