use crate::domain::SimDomainDescriptor;

/// A 2D Grid holding state for the hydraulic and sediment simulation.
/// Uses flat vectors for physical fields to ensure efficient cache locality
/// and direct compatibility with GPU compute storage buffers.
#[derive(Clone, Debug)]
pub struct GridState {
    pub width: u32,
    pub height: u32,

    // Hydrodynamic state fields (Phase 1 & 2)
    pub z_bed: Vec<f32>,     // Bed elevation (meters)
    pub h: Vec<f32>,         // Water depth (meters)
    pub u: Vec<f32>,         // Velocity X (m/s)
    pub v: Vec<f32>,         // Velocity Y (m/s)

    // Sediment & Geotechnical state fields (Phase 3)
    pub sediment_c: Vec<f32>, // Volumetric suspended sediment concentration C in [0.0, 1.0]
    pub soil_sat: Vec<f32>,   // Soil moisture / saturation W_sat in [0.0, 1.0]
    pub bedrock_z: Vec<f32>,  // Non-erodible bedrock floor elevation (meters)
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
            sediment_c: vec![0.0; size],
            soil_sat: vec![0.0; size],
            bedrock_z: vec![0.0; size],
        }
    }

    #[inline(always)]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    /// Calculates the total fluid volume in the interior simulation domain (excluding ghost halo cells).
    pub fn interior_mass(&self) -> f32 {
        let mut mass = 0.0f64;
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                mass += self.h[self.idx(x, y)] as f64;
            }
        }
        mass as f32
    }

    /// Calculates the total solid sediment volume in the interior simulation domain:
    ///
    /// Total Solid Volume = sum(z_bed) + sum(C * h / (1 - p))
    pub fn interior_sediment_mass(&self, porosity: f32) -> f32 {
        let factor = 1.0f64 / (1.0 - porosity.clamp(0.01, 0.99) as f64);
        let mut total = 0.0f64;
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let idx = self.idx(x, y);
                total += (self.z_bed[idx] as f64) + ((self.sediment_c[idx] as f64) * (self.h[idx] as f64) * factor);
            }
        }
        total as f32
    }

    /// Calculates the total water volume in the interior domain including both surface water and soil pore water:
    ///
    /// Total Water Volume = sum(h) + sum(soil_sat * min(z_bed - bedrock_z, 0.20) * porosity)
    pub fn interior_total_water_mass(&self, porosity: f32) -> f32 {
        let mut total = 0.0f64;
        let p = porosity.clamp(0.01, 0.99) as f64;
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let idx = self.idx(x, y);
                let soil_depth = (self.z_bed[idx] - self.bedrock_z[idx]).clamp(0.0, 0.20) as f64;
                let pore_water = (self.soil_sat[idx] as f64) * soil_depth * p;
                total += (self.h[idx] as f64) + pore_water;
            }
        }
        total as f32
    }

    /// Applies reflective wall boundary conditions to the 1-cell ghost halo ring.
    pub fn apply_reflective_boundaries(&mut self) {
        crate::boundary::apply_domain_boundaries(self, &crate::boundary::DomainBoundaryConfig::default(), 0.0);
    }

    /// Applies configured domain boundary conditions to the 1-cell ghost halo ring.
    pub fn apply_boundaries(&mut self, boundaries: &crate::boundary::DomainBoundaryConfig, time: f32) {
        crate::boundary::apply_domain_boundaries(self, boundaries, time);
    }
}

/// Holds two grids for double-buffered iterative operations.
#[derive(Clone, Debug)]
pub struct DoubleBufferedGrid {
    pub current: GridState,
    pub next: GridState,
    pub descriptor: SimDomainDescriptor,
    pub boundaries: crate::boundary::DomainBoundaryConfig,
    pub time: f32,
}

impl DoubleBufferedGrid {
    pub fn new(descriptor: SimDomainDescriptor) -> Self {
        let current = GridState::new(descriptor.grid_res_x, descriptor.grid_res_y);
        let next = GridState::new(descriptor.grid_res_x, descriptor.grid_res_y);
        Self {
            current,
            next,
            descriptor,
            boundaries: crate::boundary::DomainBoundaryConfig::default(),
            time: 0.0,
        }
    }

    /// Swaps the current and next buffers
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }

    /// Applies configured boundary conditions to the current grid buffer.
    pub fn apply_boundaries(&mut self) {
        crate::boundary::apply_domain_boundaries(&mut self.current, &self.boundaries, self.time);
    }

    /// Applies reflective wall boundary conditions to the current grid buffer.
    pub fn apply_reflective_boundaries(&mut self) {
        crate::boundary::apply_domain_boundaries(&mut self.current, &self.boundaries, self.time);
    }
}
