use fenrir::model::PVModel;
use fenrir::setup::*;
use fenrir::model;
use fenrir::utils::BoardTensor;
use fenrir::utils::ModelInput;
use fenrir::utils::TBoard;
use game::game::SimpleGame;
use mpi::traits::Communicator;
use mpi::traits::CommunicatorCollectives;
use mpi::traits::Destination;
use mpi::traits::Source;
use std::{path::Path};
use mpi;

/* Some MPI functionalities are hard to test with just cargo test, therefore I made this separate binary
Resulting bin is to be called from the directory where the src dir of fenrir resides */
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    println!("hello from rank {}", rank);
    let model_path = Path::new("./test_data/models").join("test_model");
    let model_config = fenrir::model::ModelConfig::load_from_toml("./config/seven_model_simple.toml").unwrap();
    let module_load_info_1 = ModuleLoadInfo::module_load_info(model_path.clone().into_os_string(), Some(0), tch::Device::Cpu, model_config.clone());
    let module_load_info_2 = ModuleLoadInfo::module_load_info(model_path.clone().into_os_string(), Some(1), tch::Device::Cpu, model_config);
    let setup_config = ModelSetupConfig::model_setup_config(true, vec![module_load_info_1, module_load_info_2]);
    let v = setup_w_mpi::<model::GeneralPVDualModel>(&setup_config).unwrap();
    world.barrier();
    if rank == 0 {
        std::fs::remove_file(&model_path).unwrap();
    }
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].0, model_path.clone());
    
    // Here we check if the modules in the two nodes are actually identical
    /* it expects input size of (_, 5, 7, 7) or (5, 7, 7)*/
    let game = SimpleGame::init_std();
    let input = TBoard::<SimpleGame>::get_pnet_input(&game, fenrir::utils::Rotation::Do(1), (tch::Kind::Float, tch::Device::Cpu));
    let input = input.get();

    let (_logit, value) = v[0].1.evaluate_t(&input, false);

    let value_data = value.double_value(&[0]);
    if rank != 0 {
        world.process_at_rank(0).send(&[value_data]);
        world.process_at_rank(0).send(&[value_data]);
    } else {
        let mut buf = vec![0.0f64; 10];
        let status = world.process_at_rank(1).receive_into(&mut buf);
        assert_eq!(status.count(<f64 as mpi::datatype::Equivalence>::equivalent_datatype()), 1);
        assert!(buf[1..].iter().all(|x| *x == 0.0));
        println!("my value {}, received {}", value_data, buf[0]);
        /* This check is about basic mpi buffer behavior. Second time comm should result in the exact same result */
        let status = world.process_at_rank(1).receive_into(&mut buf);
        assert_eq!(status.count(<f64 as mpi::datatype::Equivalence>::equivalent_datatype()), 1);
        assert!(buf[1..].iter().all(|x| *x == 0.0));
    }
}