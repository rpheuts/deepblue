pub mod params;
pub mod simulator;
pub mod init;
pub mod compute;
pub mod sync;
pub mod backend_impl;

#[cfg(test)]
mod tests;

pub use params::{GpuStepParams, BrushParams};
pub use simulator::WgpuSimulator;
pub use compute::apply_brush_to_grid;
