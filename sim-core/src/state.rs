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

    /// Calculates the total fluid volume in the interior simulation domain (excluding ghost halo cells).
    pub fn interior_mass(&self) -> f32 {
        let mut mass = 0.0;
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                mass += self.h[self.idx(x, y)];
            }
        }
        mass
    }

    /// Applies reflective wall boundary conditions to the 1-cell ghost halo ring.
    ///
    /// The physical domain consists of interior cells [1..width-2, 1..height-2].
    /// Ghost cells [x=0, x=width-1, y=0, y=height-1] mirror the interior state with
    /// normal velocity inverted, enforcing zero normal flux across domain boundaries.
    pub fn apply_reflective_boundaries(&mut self) {
        let width = self.width;
        let height = self.height;

        if width < 3 || height < 3 {
            return;
        }

        // 1. Top and Bottom ghost rows (excluding corners)
        for x in 1..(width - 1) {
            let idx_t0 = self.idx(x, 0);
            let idx_t1 = self.idx(x, 1);
            self.h[idx_t0] = self.h[idx_t1];
            self.z_bed[idx_t0] = self.z_bed[idx_t1];
            self.u[idx_t0] = self.u[idx_t1];
            self.v[idx_t0] = -self.v[idx_t1]; // Invert normal velocity

            let idx_b0 = self.idx(x, height - 1);
            let idx_b1 = self.idx(x, height - 2);
            self.h[idx_b0] = self.h[idx_b1];
            self.z_bed[idx_b0] = self.z_bed[idx_b1];
            self.u[idx_b0] = self.u[idx_b1];
            self.v[idx_b0] = -self.v[idx_b1]; // Invert normal velocity
        }

        // 2. Left and Right ghost columns (excluding corners)
        for y in 1..(height - 1) {
            let idx_l0 = self.idx(0, y);
            let idx_l1 = self.idx(1, y);
            self.h[idx_l0] = self.h[idx_l1];
            self.z_bed[idx_l0] = self.z_bed[idx_l1];
            self.u[idx_l0] = -self.u[idx_l1]; // Invert normal velocity
            self.v[idx_l0] = self.v[idx_l1];

            let idx_r0 = self.idx(width - 1, y);
            let idx_r1 = self.idx(width - 2, y);
            self.h[idx_r0] = self.h[idx_r1];
            self.z_bed[idx_r0] = self.z_bed[idx_r1];
            self.u[idx_r0] = -self.u[idx_r1]; // Invert normal velocity
            self.v[idx_r0] = self.v[idx_r1];
        }

        // 3. Four corners: diagonal reflection
        let corners = [
            (0, 0, 1, 1),
            (width - 1, 0, width - 2, 1),
            (0, height - 1, 1, height - 2),
            (width - 1, height - 1, width - 2, height - 2),
        ];
        for (gx, gy, ix, iy) in corners {
            let g_idx = self.idx(gx, gy);
            let i_idx = self.idx(ix, iy);
            self.h[g_idx] = self.h[i_idx];
            self.z_bed[g_idx] = self.z_bed[i_idx];
            self.u[g_idx] = -self.u[i_idx];
            self.v[g_idx] = -self.v[i_idx];
        }
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

    /// Applies reflective wall boundary conditions to the current grid buffer.
    pub fn apply_reflective_boundaries(&mut self) {
        self.current.apply_reflective_boundaries();
    }
}
