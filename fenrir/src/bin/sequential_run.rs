#![cfg(not(feature = "bench"))]
use fenrir::model::{GeneralPVSepModel, PVModel};
use fenrir::{load_comp_config, CompConfig};
use fenrir::agent::{load_mcts_config, MCTSConfig};
use fenrir::utils::{ActionTensor, TAction, TBoard, ModelInput};
use fenrir::replay_buffer::{BoardData, ReplayBuffer, Sampler, SimpleGameSPR};
use fenrir::self_play::{ModuleShelf, LockedShelf, Shelf, InferenceManager, self_play_new};
use fenrir::setup::{setup_wo_mpi};
use fenrir::train::{NewTrainer, ModelSyncConfig, self};
use fenrir::node::{numel_for_model, print_now};
use fenrir::duel::{duel, DuelResult};

use game::game::{GameLogic};
use game::board::{TaflBoard};
use bitboard::{BitBoard};

use color_eyre::eyre::{eyre, Context, Result};
use tch;

use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path};
use std::time::{Duration, Instant};

const SYNC_BN_STATS_EVERY: usize = 10;

fn main() -> Result<()>{

    // might be needed to correctly link cuda-related shared objects.
    let _ = tch::Cuda::cudnn_is_available();

    let argv = std::env::args().collect::<Vec<_>>();

    if argv.len() < 5 {
        eprintln!(
            "Usage: {} <config_file> <mcts_config_file> <model_dir> <data_dir>, (<initial self play game count>, <initial training step count>)",
        argv[0]);
        std::process::exit(1);
    }

    if !Path::new(&argv[1]).exists() {
    eprintln!("Config file not found: {}", argv[1]);
    std::process::exit(1);
    }

    if !Path::new(&argv[2]).exists() {
    eprintln!("Config file not found: {}", argv[2]);
    std::process::exit(1);
    }

    let config: CompConfig = load_comp_config(&argv[1]);
    let duel_mcts_config = load_mcts_config(&argv[2]);
    let model_dir = &argv[3];
    let data_dir = &argv[4];
    let init_self_play_count: usize = if let Some(i) = argv.get(5) {
        i.parse().unwrap()
    } else {
        0
    };
    let init_training_step_count: usize = if let Some(i) = argv.get(6) {
        i.parse().unwrap()
    } else {
        0
    };

    sequential_run::<GeneralPVSepModel, SimpleGameSPR, _>(
        &config,
        duel_mcts_config,
        model_dir,
        data_dir,
        init_training_step_count,
        init_self_play_count)?;
    Ok(())
}

fn sequential_run<P: PVModel + Send + 'static, D: BoardData, A: AsRef<Path>>(
    config: &CompConfig,
    duel_mcts_config: MCTSConfig,
    model_dir: A,
    data_dir: A,
    init_training_step_count: usize,
    init_self_play_count: usize,
) -> Result<()> 
where
ReplayBuffer<D>: Sampler,
TBoard<<D as BoardData>::G>: ModelInput<D::G>,
TAction<<D::G as GameLogic>::B>: ActionTensor,
TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display,
<<D::G as GameLogic>::B as BitBoard>::Movement: PartialEq{

    assert!(!config.fenrir_config.use_mpi);
    let start = Instant::now();

    let loaded_modules = setup_wo_mpi::<P>(&config.setup_config)?;
    let (table,name_lookup) = config.setup_config.create_lookup_table(None, loaded_modules)?;

    let mut shelf = ModuleShelf::module_shelf(table, name_lookup);
    let replay_buffer = ReplayBuffer::<D>::new(config.fenrir_config.replay_buffer_capacity);
    let (_numel, capacity) = numel_for_model(&shelf.get_group(0).unwrap().get(0).unwrap().1);
    let mut storage = Vec::<u8>::with_capacity(capacity);

    let mut training_step_count: usize = init_training_step_count;
    let mut self_play_game_count: usize = init_self_play_count;

    // fs-related
    std::fs::create_dir_all(&model_dir)?;
    std::fs::create_dir_all(&data_dir)?;
    let loss_data_path = data_dir.as_ref().to_path_buf().join(format!("{}-loss.dat", config.name));
    let self_play_data_path = data_dir.as_ref().to_path_buf().join(format!("{}-self-play.dat", config.name));
    let mut self_play_file = std::fs::File::create(&self_play_data_path).wrap_err("Could not create file for self-play data storage")?;
    writeln!(self_play_file, "attacker defender draw")?;
    // drop(self_play_file);
    let model_store_path = model_dir.as_ref().to_path_buf().join(format!("{}.pv", config.name));
    let mut loss_data_file = std::fs::File::create(&loss_data_path).wrap_err("Could not create file for loss data storage")?;
    writeln!(loss_data_file, "cross-entropy mse total")?;

    let checkpoint_path = model_dir.as_ref().to_path_buf().join("tmp.pv");

    fn check_shutdown<P: PVModel + Send, A: AsRef<Path>>(shelf: &ModuleShelf<P, P>, start: &Instant, model_store_path: A, config: &CompConfig) -> bool {
        let elapsed = start.elapsed();
        if elapsed > Duration::from_secs_f32(3600.0 * config.fenrir_config.run_time_hr) {

            let mut f = File::create(&model_store_path).unwrap();
            shelf.write_to_stream(0, &mut f).unwrap();
            print_now(&format!("Current best model was saved at {:?}", model_store_path.as_ref()));
            true
        } else {
            false
        }
    } 
 
    loop {

        if check_shutdown(&shelf, &start, &model_store_path, config) {
            break;
        }

        let mut locked_shelf = LockedShelf::<P>::convert_from_shelf(shelf);

        let (manager, mut request_senders) = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut locked_shelf, config.fenrir_config.inference_bs);
        let request_sender = request_senders.remove(0);

        // data-generation
        self_play_new::<P,D, &mut std::fs::File>(
            manager,
            request_sender,
            config.fenrir_config.n_self_play_games,
            &config.mcts_config,
            &replay_buffer,
            Some(&mut self_play_file)
        )?;

        self_play_game_count += config.fenrir_config.n_self_play_games;
        print_now(&format!("Data generation phase complete. Total of {} games have been played", self_play_game_count));

        // keep the pre-training state so that later we can use for evaluation
        storage.clear();
        locked_shelf.write_to_stream(0, &mut storage)?;

        shelf = LockedShelf::<P>::convert_into_shelf(locked_shelf);

        // Train
        let mut trainer = NewTrainer::<P, &mut ModuleShelf<P,P>>::new::<tch::nn::Sgd>(
            &mut shelf,
            tch::nn::sgd(config.fenrir_config.momentum, 0.0f64, config.fenrir_config.weight_decay, false),
            (config.fenrir_config.learning_rate_schedule)(training_step_count as usize),
            config.fenrir_config.weight_decay,
            config.fenrir_config.train_bs,
            ModelSyncConfig::new(train::SumOrAve::Ave)
        )?;

        trainer.train(config.fenrir_config.n_training_step_per_cycle, SYNC_BN_STATS_EVERY, &replay_buffer, &mut rand::thread_rng())?;
        trainer.flush_loss_record(&mut loss_data_file);
        training_step_count += trainer.step_count;
        drop(trainer);

        print_now(&format!("Training phase complete: total of {} steps", training_step_count));
        
        let mut f = File::create(&checkpoint_path).unwrap();
        shelf.write_to_stream(0, &mut f).unwrap();

        if check_shutdown(&shelf, &start, &model_store_path, config) {
            break;
        }

        // Evaluation
        assert_eq!(shelf.table.len(), 1); // There should be only one model loaded right now
        let n_replicas = shelf.table[0].len();
        assert!(n_replicas > 0);
        if n_replicas == 1 {
            return Err(eyre!("shelf contains only one replica - this is not sufficient for duel. Make sure to load at least 2 replicas from the start"));
        }
        let split = n_replicas / 2;
        let mut tmp = std::mem::take(&mut shelf.table);
        let mut tmp = tmp.remove(0);
        let splitted = tmp.split_off(split);
        shelf.table = vec![tmp, splitted];
        assert_eq!(shelf.table.len(), 2);
        let mut cursor = Cursor::new(&mut storage);
        shelf.update_modules_from_stream(0, &mut cursor)?;
        let mut locked_shelf = LockedShelf::<P>::convert_from_shelf(shelf);
        let (manager, mut request_senders) = InferenceManager::<'_, P, &'_ mut LockedShelf<P>>::new(&mut locked_shelf, config.fenrir_config.inference_bs);
        assert_eq!(request_senders.len(), 2);

        let champion_rs = request_senders.remove(0);
        let challenger_rs = request_senders.remove(0);

        let duel_result: DuelResult = duel::<P, D>(manager, champion_rs, challenger_rs, config.fenrir_config.n_games_per_tournament, &duel_mcts_config, &duel_mcts_config)?;
        let dr: (f32, f32, f32, f32, f32, f32) = (duel_result.0 as f32,duel_result.1 as f32,duel_result.2 as f32,duel_result.3 as f32,duel_result.4 as f32,duel_result.5 as f32);

        // win-rate for challenger when it is the attacker
        let chal_wr_when_def = (dr.0 - dr.1 - 0.5 * dr.2) / dr.0; // This formula might need tweaking
        // win-rate for challenger when defender
        let chal_wr_when_att = (dr.3 - dr.4 - 0.5 * dr.5) / dr.3;

        let msg: String = "Evaluation of a checkpoint completed\n".to_string() + 
        &format!("challenger's win rate | draw count as defender: {}|{}\n", chal_wr_when_def, dr.2) + 
        &format!("challenger's win rate | draw count as attacker: {}|{}\n", chal_wr_when_att, duel_result.5);
        print_now(&msg);

        if chal_wr_when_att > config.fenrir_config.model_update_threshold 
            && chal_wr_when_def > config.fenrir_config.model_update_threshold {
            print_now("Updating the model..");
            storage.clear();
            locked_shelf.write_to_stream(1, &mut storage)?;
            let mut cursor = Cursor::new(&mut storage);
            locked_shelf.update_modules_from_stream(0, &mut cursor)?;
        } else {
            print_now("Skipping the update..");
            
            // revert the change
            storage.clear();
            locked_shelf.write_to_stream(0, &mut storage)?;
            let mut cursor = Cursor::new(&mut storage);
            locked_shelf.update_modules_from_stream(1, &mut cursor)?;
        }

        shelf = locked_shelf.convert_into_shelf();
        let mut tmp = std::mem::take(&mut shelf.table);
        assert_eq!(tmp.len(), 2);
        let mut tmp1 = tmp.remove(0);
        tmp1.append(&mut tmp[0]);
        shelf.table = vec![tmp1];

    }
    Ok(())
}
