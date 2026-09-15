pub mod domain;
pub mod state;
pub mod boundary;
pub mod solver;
pub mod backend;
pub mod scenario;

pub use backend::{SimulationBackend, CpuSimulator};
pub use boundary::{EdgeBoundary, DomainBoundaryConfig};
pub use scenario::Scenarios;

#[cfg(test)]
mod tests;
