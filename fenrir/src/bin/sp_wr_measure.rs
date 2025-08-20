use color_eyre::eyre::Result;
#[allow(unused_imports)]
use fenrir::model::{GeneralPVDualModel, GeneralPVSepModel};
use fenrir::replay_buffer::SimpleGameSPR;
use fenrir::setup::{ModelSetupConfig, setup_wo_mpi};
use fenrir::agent::MCTSConfig;
use fenrir::self_play::{self_play_wo_rec, InferenceManager, LockedShelf, ModuleShelf};
const BS: usize = 8;
const NUM_GAMES: usize = 100;

fn main() -> Result<()>{

    type P = GeneralPVSepModel;
    type D = SimpleGameSPR;

    // might be needed to correctly link cuda-related shared objects.
    let _ = tch::Cuda::cudnn_is_available();

    let argv = std::env::args().collect::<Vec<_>>(); // [_, comp_config]
    if argv.len() != 2 {
        eprintln!("Usage: {} <setup config file>", argv[0]);
    }

    let setup_config: ModelSetupConfig = ModelSetupConfig::parse_toml(&argv[1]).unwrap();
    let loaded_modules = setup_wo_mpi::<P>(&setup_config).unwrap();

    println!("setup config: \n{:?}", setup_config);
    println!("Att Def Draw");

    let mut mcts_config = MCTSConfig::default();
    let num_sims: [usize; 5] = [100, 200, 400, 800, 1600];

    let (table, name_lookup) = ModelSetupConfig::create_lookup_table(&setup_config, None, loaded_modules).unwrap();

    let mut shelf = ModuleShelf::module_shelf(table, name_lookup);

    for n_sim in num_sims.iter() {

        println!("Number of simulations per move: {}", n_sim);
        mcts_config.n_sim = *n_sim;
        let mut locked_shelf = LockedShelf::convert_from_shelf(shelf);
        let (manager, mut request_senders)
        = InferenceManager::<GeneralPVSepModel, _>::new(&mut locked_shelf, BS);
        assert_eq!(request_senders.len(), 1);
        let request_sender = request_senders.remove(0);
        
        self_play_wo_rec::<P, D, _>(
            manager,
            request_sender,
            NUM_GAMES,
            &mcts_config,
            &mut std::io::stdout()
        )?;

        shelf = LockedShelf::convert_into_shelf(locked_shelf);
    }
    Ok(())
}