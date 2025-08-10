use crate::model::PVModel;
use crate::node::{Node, SelfPlayNode, TrainNode, TestNode};
use crate::replay_buffer::{BoardData, ReplayBuffer, Sampler};
use crate::utils::{ActionTensor, ModelInput, TAction, TBoard};
use crate::{CompConfig, MpiConfig};

use bincode::Encode;
use game::board::TaflBoard;
use game::game::GameLogic;
#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::Rank;
use mpi::{point_to_point::*, Tag};
use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use mpi::MpiError;
use mpi::datatype::{DynBufferMut, Equivalence};

pub fn run_cnt<P: PVModel + Send, D: BoardData + Encode> (config: CompConfig, mpi_config: MpiConfig)
where 
TBoard<D::G>: ModelInput<D::G>,
TAction<<D::G as GameLogic>::B>: ActionTensor,
TaflBoard<<D::G as GameLogic>::B>: std::fmt::Display,
D: Send + Sync,
ReplayBuffer<D>: Sampler
{
    let (universe, threading) = mpi::initialize_with_threading(mpi::Threading::Multiple).unwrap();
    let world = universe.world();
    let rank = world.rank();
    
    if rank == mpi_config.test {
        println!("Hello from rank {}", rank);
        let mut node = TestNode::<P, D>::init(config);
        node.run(&mpi_config);
    } else if rank == mpi_config.train {
        println!("Hello from rank {}", rank);
        let mut node = TrainNode::<P, D>::init(config);
        node.run(&mpi_config);
    } else if mpi_config.self_play.contains(&rank) {
        println!("Hello from rank {}", rank);
        let mut node = SelfPlayNode::<P, D>::init(config);
        node.run(&mpi_config);
    }

}
