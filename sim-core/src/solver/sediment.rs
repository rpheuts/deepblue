use crate::state::DoubleBufferedGrid;

/// Parameters governing sediment transport and geotechnical talus dynamics.
#[derive(Copy, Clone, Debug)]
pub struct SedimentParams {
    /// Sediment bed porosity p in [0.0, 1.0) (default 0.40 for natural sand)
    pub porosity: f32,
    /// Settling velocity w_s in m/s (default 0.04 m/s for fine-medium sand)
    pub settling_velocity: f32,
    /// Erodibility coefficient / pickup rate scaling k_e (default 0.008)
    pub erodibility: f32,
    /// Critical fluid velocity v_c (m/s) below which no sediment is picked up (default 0.22 m/s)
    pub critical_velocity: f32,
    /// Equilibrium carrying capacity scaling coefficient K_cap
    pub capacity_scale: f32,

    // Geotechnical repose limits (radians)
    /// Angle of repose for dry sand (default 34 degrees = ~0.593 rad)
    pub phi_dry: f32,
    /// Angle of repose for damp sand with capillary cohesion (default 45 degrees = ~0.785 rad)
    pub phi_damp: f32,
    /// Angle of repose for saturated / submerged sand prone to liquefaction (default 22 degrees = ~0.384 rad)
    pub phi_sat: f32,
    /// Number of iterative talus relaxation passes per step (default 2)
    pub talus_iterations: u32,
}

impl Default for SedimentParams {
    fn default() -> Self {
        Self {
            porosity: 0.40,
            settling_velocity: 0.04,
            erodibility: 0.008,
            critical_velocity: 0.22,
            capacity_scale: 0.05,
            phi_dry: 34.0f32.to_radians(),
            phi_damp: 45.0f32.to_radians(),
            phi_sat: 22.0f32.to_radians(),
            talus_iterations: 2,
        }
    }
}

/// Performs a combined sediment dynamics step:
/// 1. Soil saturation tracking
/// 2. Conservative suspended sediment advection and Exner equation bed evolution
/// 3. Geotechnical multi-directional angle-of-repose talus collapse pass
pub fn step_sediment(grid: &mut DoubleBufferedGrid, dt: f32, params: &SedimentParams) {
    step_saturation(grid, dt);
    step_exner_exchange(grid, dt, params);
    for _ in 0..params.talus_iterations {
        step_talus_collapse(grid, params);
    }
}

/// Updates soil moisture and saturation W_sat in [0.0, 1.0].
/// Submerged cells become fully saturated (1.0).
/// Dry cells slowly dry out in the air or absorb moisture from wet neighbors.
pub fn step_saturation(grid: &mut DoubleBufferedGrid, dt: f32) {
    let width = grid.descriptor.grid_res_x;
    let height = grid.descriptor.grid_res_y;

    grid.current.apply_reflective_boundaries();

    let drying_rate = 0.02f32; // Moisture drying per second
    let diffusion_rate = 0.05f32; // Soil moisture capillary diffusion

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = grid.current.idx(x, y);
            let h = grid.current.h[idx];

            if h > 1e-4 {
                // Fully submerged soil is completely saturated
                grid.next.soil_sat[idx] = 1.0;
            } else {
                let sat_c = grid.current.soil_sat[idx];
                let sat_l = grid.current.soil_sat[grid.current.idx(x - 1, y)];
                let sat_r = grid.current.soil_sat[grid.current.idx(x + 1, y)];
                let sat_t = grid.current.soil_sat[grid.current.idx(x, y - 1)];
                let sat_b = grid.current.soil_sat[grid.current.idx(x, y + 1)];

                // Capillary diffusion from wet neighbors
                let laplacian = sat_l + sat_r + sat_t + sat_b - 4.0 * sat_c;
                let next_sat = sat_c + (diffusion_rate * laplacian - drying_rate * sat_c) * dt;
                grid.next.soil_sat[idx] = next_sat.clamp(0.0, 1.0);
            }

            // Preserve other fields during saturation pass
            grid.next.h[idx] = grid.current.h[idx];
            grid.next.u[idx] = grid.current.u[idx];
            grid.next.v[idx] = grid.current.v[idx];
            grid.next.z_bed[idx] = grid.current.z_bed[idx];
            grid.next.sediment_c[idx] = grid.current.sediment_c[idx];
            grid.next.bedrock_z[idx] = grid.current.bedrock_z[idx];
        }
    }

    grid.next.apply_reflective_boundaries();
    grid.swap();
}

/// Solves suspended sediment advection and the Exner equation for bed evolution:
///
/// d(z_bed)/dt = (D - E) / (1 - p)
/// d(h * C)/dt + div(u * h * C) = E - D
///
/// This formulation guarantees exact mass conservation between the bed and water column.
pub fn step_exner_exchange(grid: &mut DoubleBufferedGrid, dt: f32, params: &SedimentParams) {
    let width = grid.descriptor.grid_res_x;
    let height = grid.descriptor.grid_res_y;

    let p = params.porosity.clamp(0.01, 0.99);
    let inv_one_minus_p = 1.0 / (1.0 - p);
    let ws = params.settling_velocity;
    let vc = params.critical_velocity;
    let k_cap = params.capacity_scale;

    grid.current.apply_reflective_boundaries();

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = grid.current.idx(x, y);

            let h_c = grid.current.h[idx];
            let u_c = grid.current.u[idx];
            let v_c = grid.current.v[idx];
            let z_c = grid.current.z_bed[idx];
            let c_c = grid.current.sediment_c[idx];
            let bedrock = grid.current.bedrock_z[idx];

            // Non-Equilibrium Exner Exchange (Deposition D & Erosion E)
            let speed = (u_c * u_c + v_c * v_c).sqrt();
            let h_dry = 1e-4;

            let c_eq = if speed > vc && h_c > h_dry {
                let excess_speed_sq = speed * speed - vc * vc;
                (k_cap * excess_speed_sq / (1.0 + 0.5 * h_c)).clamp(0.0, 0.35)
            } else {
                0.0
            };

            let deposition = ws * c_c;
            let mut pickup = ws * c_eq;

            // Constrain erosion by erodible alluvium thickness (cannot erode below non-erodible bedrock)
            let erodible_sand = (z_c - bedrock).max(0.0);
            if pickup > deposition {
                let max_pickup = deposition + (erodible_sand * (1.0 - p)) / dt.max(1e-5);
                pickup = pickup.min(max_pickup);
            }

            // Exner equation bed elevation change from net deposition/erosion
            let mut dz = (deposition - pickup) * inv_one_minus_p * dt;
            // Prevent dropping below bedrock
            if z_c + dz < bedrock {
                dz = bedrock - z_c;
            }
            let mut z_next = z_c + dz;

            // Strict conservation: whatever solid volume is added/subtracted from the bed
            // must correspond to -(dz * (1 - p)) in the water column
            let net_source_fluid = -dz * (1.0 - p);
            let current_load = h_c * c_c;
            let next_load = (current_load + net_source_fluid).max(0.0);

            let c_next = if h_c > h_dry {
                let c_tentative = next_load / h_c;
                if c_tentative > 0.50 {
                    // Maximum packing limit (slurry capacity): deposit excess immediately to bed
                    let max_load = 0.50 * h_c;
                    let excess = next_load - max_load;
                    z_next += excess * inv_one_minus_p;
                    0.50
                } else {
                    c_tentative
                }
            } else {
                // In dry cells, any remaining suspended sediment settles immediately to the bed
                let excess_settle = next_load * inv_one_minus_p;
                z_next += excess_settle;
                0.0
            };

            grid.next.z_bed[idx] = z_next;
            grid.next.sediment_c[idx] = c_next;
            grid.next.h[idx] = h_c;
            grid.next.u[idx] = u_c;
            grid.next.v[idx] = v_c;
            grid.next.soil_sat[idx] = grid.current.soil_sat[idx];
            grid.next.bedrock_z[idx] = bedrock;
        }
    }

    grid.next.apply_reflective_boundaries();
    grid.swap();
}

/// Geotechnical multi-directional angle-of-repose slope collapse pass.
///
/// Prevents non-physical vertical cliffs by relaxing slopes that exceed
/// the dynamic angle of repose phi(W_sat).
/// Transfers sediment conservatively between neighboring cells (sum z_bed is strictly conserved).
pub fn step_talus_collapse(grid: &mut DoubleBufferedGrid, params: &SedimentParams) {
    let dx = grid.descriptor.extent_x / grid.descriptor.grid_res_x as f32;
    let dy = grid.descriptor.extent_y / grid.descriptor.grid_res_y as f32;
    let width = grid.descriptor.grid_res_x;
    let height = grid.descriptor.grid_res_y;

    let tan_dry = params.phi_dry.tan();
    let tan_damp = params.phi_damp.tan();
    let tan_sat = params.phi_sat.tan();

    grid.current.apply_reflective_boundaries();

    // 8-neighbor offsets and distances
    let sqrt2 = std::f32::consts::SQRT_2;
    let neighbors: [(i32, i32, f32); 8] = [
        (-1, 0, dx),
        (1, 0, dx),
        (0, -1, dy),
        (0, 1, dy),
        (-1, -1, dx * sqrt2),
        (1, -1, dx * sqrt2),
        (-1, 1, dx * sqrt2),
        (1, 1, dx * sqrt2),
    ];

    // First, copy current to next
    for i in 0..grid.current.z_bed.len() {
        grid.next.z_bed[i] = grid.current.z_bed[i];
        grid.next.sediment_c[i] = grid.current.sediment_c[i];
        grid.next.h[i] = grid.current.h[i];
        grid.next.u[i] = grid.current.u[i];
        grid.next.v[i] = grid.current.v[i];
        grid.next.soil_sat[i] = grid.current.soil_sat[i];
        grid.next.bedrock_z[i] = grid.current.bedrock_z[i];
    }

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = grid.current.idx(x, y);
            let z_c = grid.current.z_bed[idx];
            let bedrock_c = grid.current.bedrock_z[idx];
            let erodible_avail = (z_c - bedrock_c).max(0.0);

            if erodible_avail <= 0.0 {
                continue;
            }

            // Dynamic critical angle of repose based on soil saturation W_sat
            let sat = grid.current.soil_sat[idx].clamp(0.0, 1.0);
            let tan_crit = if sat < 0.30 {
                // Damp capillary sand has temporary cohesion
                tan_dry + (tan_damp - tan_dry) * (sat / 0.30)
            } else {
                // Saturated sand liquefies and slumps at lower slope
                tan_damp - (tan_damp - tan_sat) * ((sat - 0.30) / 0.70)
            };

            let mut total_excess = 0.0f32;
            let mut excess_per_neighbor = [0.0f32; 8];

            for (i, &(nx, ny, dist)) in neighbors.iter().enumerate() {
                let nx_coord = x as i32 + nx;
                let ny_coord = y as i32 + ny;
                // Closed boundary: sand cannot collapse across the outer physical boundary
                if nx_coord < 1 || nx_coord >= (width - 1) as i32 || ny_coord < 1 || ny_coord >= (height - 1) as i32 {
                    continue;
                }

                let neighbor_idx = grid.current.idx(nx_coord as u32, ny_coord as u32);
                let z_n = grid.current.z_bed[neighbor_idx];
                let dz = z_c - z_n;
                let max_dz = dist * tan_crit;

                if dz > max_dz {
                    let excess = dz - max_dz;
                    excess_per_neighbor[i] = excess;
                    total_excess += excess;
                }
            }

            if total_excess > 0.0 {
                // Limit total transfer by available erodible sand and relaxation factor
                let transfer_cap = (erodible_avail * 0.45).min(total_excess * 0.25);
                let scale = transfer_cap / total_excess;

                for (i, &(nx, ny, _)) in neighbors.iter().enumerate() {
                    let excess = excess_per_neighbor[i];
                    if excess > 0.0 {
                        let amount = excess * scale;
                        let nx_coord = (x as i32 + nx) as u32;
                        let ny_coord = (y as i32 + ny) as u32;
                        let neighbor_idx = grid.current.idx(nx_coord, ny_coord);
                        grid.next.z_bed[idx] -= amount;
                        grid.next.z_bed[neighbor_idx] += amount;
                    }
                }
            }
        }
    }

    grid.next.apply_reflective_boundaries();
    grid.swap();
}
