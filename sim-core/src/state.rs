use crate::domain::SimDomainDescriptor;

/// A simple 2D Grid holding state for the simulation.
/// Uses a flat vector for data to be GPU friendly in the future.
#[derive(Clone, Debug)]
pub struct GridState {
    pub width: u32,
    pub height: u32,
    
    // Physical state fields
    pub z_bed: Vec<f32>,     // Elevation map
    pub h: Vec<f32>,         // Water depth
    pub u: Vec<f32>,         // Velocity X (or discharge uh)
    pub v: Vec<f32>,         // Velocity Y (or discharge vh)
}

impl GridState {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            z_bed: vec![0.0; size],
            h: vec![0.0; size],
            u: vec![0.0; size],
            v: vec![0.0; size],
        }
    }

    #[inline(always)]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }
}

/// Holds two grids for double-buffered iterative operations.
#[derive(Clone, Debug)]
pub struct DoubleBufferedGrid {
    pub current: GridState,
    pub next: GridState,
    pub descriptor: SimDomainDescriptor,
}

impl DoubleBufferedGrid {
    pub fn new(descriptor: SimDomainDescriptor) -> Self {
        let current = GridState::new(descriptor.grid_res_x, descriptor.grid_res_y);
        let next = GridState::new(descriptor.grid_res_x, descriptor.grid_res_y);
        Self {
            current,
            next,
            descriptor,
        }
    }

    /// Swaps the current and next buffers
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}
