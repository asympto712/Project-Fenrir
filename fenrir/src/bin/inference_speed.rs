#![cfg(feature = "bench")]

use fenrir::{
    agent::MCTSConfig,
    model::GeneralPVSepModel,
    replay_buffer::SimpleGameSPR,
    self_play::{self_play_bench, InferenceManager, LockedShelf, ModuleShelf},
    setup::{setup_wo_mpi, ModelSetupConfig},
};
use itertools::iproduct;

const NUM_GAMES: usize = 10;

fn main() {

    // might be needed to correctly link cuda-related shared objects.
    let _ = tch::Cuda::cudnn_is_available();

    let argv = std::env::args().collect::<Vec<_>>(); // [_, comp_config]
    if argv.len() != 2 {
        eprintln!("Usage: {} <setup config file>", argv[0]);
    }

    let setup_config: ModelSetupConfig = ModelSetupConfig::parse_toml(&argv[1]).unwrap();
    let loaded_modules = setup_wo_mpi::<GeneralPVSepModel>(&setup_config).unwrap();

    println!("setup config: \n{:?}", setup_config);

    let mut mcts_config = MCTSConfig::default();
    let num_sims: [usize; 5] = [100, 200, 400, 800, 1600];
    let batch_sizes: [usize; 5] = [1,2,4,8,16];

    let (table, name_lookup) = ModelSetupConfig::create_lookup_table(&setup_config, None, loaded_modules).unwrap();

    let mut shelf = ModuleShelf::module_shelf(table, name_lookup);

    for (n_sim, bs) in iproduct!(num_sims.iter(), batch_sizes.iter()) {

        println!("Number of simulations per move: {}, Batch size: {}", n_sim, bs);
        mcts_config.n_sim = *n_sim;
        let mut locked_shelf = LockedShelf::convert_from_shelf(shelf);
        let (manager, mut request_senders)
        = InferenceManager::<GeneralPVSepModel, _>::new(&mut locked_shelf, *bs);
        assert_eq!(request_senders.len(), 1);
        let request_sender = request_senders.remove(0);
        self_play_bench::<GeneralPVSepModel, SimpleGameSPR, _>(
            manager,
            request_sender,
            NUM_GAMES,
            &mcts_config,
            &mut std::io::stdout()
        ).unwrap();

        shelf = LockedShelf::convert_into_shelf(locked_shelf);

    }

}