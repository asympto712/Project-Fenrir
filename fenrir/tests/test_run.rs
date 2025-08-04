use fenrir::load_comp_config;
use fenrir::model::{GeneralPVDualModel, ModelConfig, PVModel};
use fenrir::replay_buffer::{GameSPR, ReplayBuffer};
use fenrir::run::{now_into_filename, setup, ModelSetupConfig, ModuleLoadInfo};
use fenrir::self_play::{ModuleShelf, LockedShelf, Shelf, InferenceManager, self_play_new};
use fenrir::CompConfig;
use fenrir::train::Trainer;

use std::io::Write;
use std::path::{PathBuf, Path};

use tch::nn::VarStore;
use tch::Device;

use tempfile::tempfile;

#[test]
fn test_run_wo_mpi_sequential() {

    type P = GeneralPVDualModel;
    type D = GameSPR;

    std::fs::create_dir_all(Path::new("./test_data").join("models")).unwrap();

    let path = Path::new("./test_data").join("test_config1.toml");
    let comp_config: CompConfig = load_comp_config(path);
    assert!(!comp_config.fenrir_config.use_mpi);

    let path = Path::new("./test_data").join("eleven_model_simple.toml");
    let model_config = ModelConfig::load_from_toml(path).unwrap();

    // Create a new model
    let vs = VarStore::new(Device::Cpu);
    let _new_model = <P as PVModel>::new(&vs.root(), &model_config);
    let path: PathBuf = Path::new("./test_data").join("models").join(&now_into_filename());
    vs.save(&path).unwrap();

    let module_load_infos = vec![
        ModuleLoadInfo::module_load_info(
            path.clone().to_str().unwrap().to_string().into(),
            None,
            Device::Cpu,
            model_config.clone()
        )
    ];

    let model_setup_config = ModelSetupConfig::model_setup_config(false, module_load_infos);
    let loaded_modules = setup::<P>(&model_setup_config).unwrap();
    let (table, name_lookup) = model_setup_config.create_lookup_table(None, loaded_modules).unwrap();

    let mut shelf = ModuleShelf::module_shelf(table, name_lookup);
    let replay_buffer = ReplayBuffer::<D>::new(comp_config.fenrir_config.replay_buffer_capacity);

    let mut temp_file = tempfile().unwrap();

    dbg!("config load, model setup complete");

    for i in 0..1 {

        let mut locked_shelf = LockedShelf::<P>::convert_from_shelf(shelf);

        if i != 0 {
            locked_shelf.update_modules_from_stream(0, &mut temp_file).unwrap();
        }

        let (manager, request_senders) = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut locked_shelf, comp_config.fenrir_config.mini_batch_size);

        dbg!("self play phase");
        self_play_new::<P,D>(
            manager,
            request_senders[0].clone(),
            comp_config.fenrir_config.n_self_play_games,
            &comp_config.mcts_config,
            &replay_buffer
        ).unwrap();

        shelf = LockedShelf::<P>::convert_into_shelf(locked_shelf);
        let mut trainer = Trainer::<P>::new::<tch::nn::Sgd>(
            shelf.get_group_mut(0).expect("shelf is empty"),
            tch::nn::sgd(comp_config.fenrir_config.momentum, 0.0f64,comp_config.fenrir_config.weight_decay, false),
            (comp_config.fenrir_config.learning_rate_schedule)(0),
            comp_config.fenrir_config.weight_decay,
            comp_config.fenrir_config.mini_batch_size,
            comp_config.model_sync_config
        ).unwrap();

        trainer.train::<D, _>(comp_config.fenrir_config.n_training_step_per_cycle, comp_config.fenrir_config.n_training_step_per_cycle, &replay_buffer).unwrap();
        temp_file.flush().unwrap();
        trainer.save_to_stream(&mut temp_file).unwrap();

    }
}
