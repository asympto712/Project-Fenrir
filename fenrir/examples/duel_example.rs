use fenrir::{
    duel::{duel},
    model::{GeneralPVDualModel, ModelConfig, PVModel},
    replay_buffer::{SimpleGameSPR},
    self_play::{InferenceManager, LockedShelf, ModuleShelf},
    setup::{ModelSetupConfig, ModuleLoadInfo, setup},
    {CompConfig, load_comp_config},
};
use tch::{Device, nn::VarStore};
use color_eyre::eyre;

use std::path::{Path, PathBuf};


#[allow(dead_code)]
fn main() -> eyre::Result<()> {

    type P = GeneralPVDualModel;
    type D = SimpleGameSPR;

    let path = Path::new("./config").join("comp_config_test.toml");
    let comp_config: CompConfig = load_comp_config(path);
    assert!(!comp_config.fenrir_config.use_mpi);

    let path = Path::new("./config").join("seven_model_simple.toml");
    let model_config = ModelConfig::load_from_toml(path).unwrap();

    // Create a new model
    let vs_1 = VarStore::new(Device::Cpu);
    let _new_model = <P as PVModel>::new(&vs_1.root(), &model_config);
    let model_path1: PathBuf = Path::new("./test_data").join("models").join("player_1");
    vs_1.save(&model_path1).unwrap();

    // Create a new model
    let vs_2 = VarStore::new(Device::Cpu);
    let _new_model = <P as PVModel>::new(&vs_2.root(), &model_config);
    let model_path2: PathBuf = Path::new("./test_data").join("models").join("player_2");
    vs_2.save(&model_path2).unwrap();

    let module_load_infos = vec![
        ModuleLoadInfo::module_load_info(
            model_path1.clone().to_str().unwrap().to_string().into(),
            None,
            Device::Cpu,
            model_config.clone()
        ),
        ModuleLoadInfo::module_load_info(
            model_path2.clone().to_str().unwrap().to_string().into(),
            None,
            Device::Cpu,
            model_config.clone())
    ];

    let model_setup_config = ModelSetupConfig::model_setup_config(false, module_load_infos);
    let loaded_modules = setup::<P>(&model_setup_config).unwrap();
    let (table, name_lookup) = model_setup_config.create_lookup_table(None, loaded_modules).unwrap();

    let shelf = ModuleShelf::module_shelf(table, name_lookup);
    let mut locked_shelf = LockedShelf::<P>::convert_from_shelf(shelf);
    
    std::fs::remove_file(model_path1).unwrap();
    std::fs::remove_file(model_path2).unwrap();

    let (manager, mut req_senders) 
    = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut locked_shelf, comp_config.fenrir_config.inference_bs);

    assert_eq!(req_senders.len(), 2);
    let sender2 = req_senders.remove(1);
    let sender1 = req_senders.remove(0);

    let duel_res = duel::<P, D>(
        manager,
        sender1,
        sender2,
        comp_config.fenrir_config.n_games_per_tournament,
        &comp_config.mcts_config,
    &comp_config.mcts_config
    )?;

    println!("{:?}", duel_res);

    Ok(())
}
