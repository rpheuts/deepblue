use crate::state::GridState;

/// Boundary condition behavior applied to a domain perimeter edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeBoundary {
    /// Zero normal flux reflective wall (Neumann condition):
    /// u_normal = -u_interior, u_tangential = u_interior.
    SolidWall,

    /// Radiation / absorption boundary allowing fluid to drain out freely
    /// without reflecting artificial waves back into the domain.
    /// `absorption_rate` controls damping strength (0.0 to 1.0, e.g. 0.85).
    OpenOutflow {
        absorption_rate: f32,
    },

    /// Constant inflow boundary (e.g. wide river entering domain).
    ConstantInflow {
        target_depth: f32,
        inflow_velocity: f32,
    },

    /// Oscillating wave generator simulating rhythmic ocean swell, surges, and tides.
    WaveGenerator {
        /// Base still-water sea elevation (meters)
        base_elevation: f32,
        /// Wave crest amplitude (meters)
        wave_amplitude: f32,
        /// Period between wave crests (seconds)
        wave_period: f32,
        /// Onshore surge velocity at wave crest (m/s)
        surge_speed: f32,
        /// Tidal elevation swing amplitude (meters)
        tide_amplitude: f32,
        /// Full tidal cycle period (seconds)
        tide_period: f32,
    },
}

impl EdgeBoundary {
    /// Evaluates water surface elevation (eta) and normal surge velocity for time-dependent boundaries.
    pub fn evaluate_wave(&self, time: f32) -> (f32, f32) {
        match *self {
            EdgeBoundary::WaveGenerator {
                base_elevation,
                wave_amplitude,
                wave_period,
                surge_speed,
                tide_amplitude,
                tide_period,
            } => {
                let pi = std::f32::consts::PI;
                let tide_phase = if tide_period > 1e-4 {
                    (2.0 * pi * time / tide_period).sin()
                } else {
                    0.0
                };
                let tide_z = tide_amplitude * tide_phase;

                let phase = if wave_period > 1e-4 {
                    (time / wave_period).rem_euclid(1.0)
                } else {
                    0.0
                };

                // Asymmetric coastal surge profile:
                // 1. Long sustained surge: 40% of cycle (e.g. 6.0s at T=15s)
                // 2. Receding backwash: 30% of cycle (e.g. 4.5s at T=15s)
                // 3. Calm inter-surge interval: 30% of cycle (e.g. 4.5s at T=15s)
                let wave_surge = if phase < 0.40 {
                    let s = phase / 0.40;
                    (s * pi).sin().powf(1.2)
                } else {
                    let r = (phase - 0.40) / 0.60;
                    if r < 0.50 {
                        -0.35 * ((r / 0.50) * pi).sin().powf(1.2)
                    } else {
                        0.0
                    }
                };

                let target_eta = base_elevation + tide_z + wave_amplitude * wave_surge;
                let normal_velocity = if wave_surge > 0.0 {
                    surge_speed * wave_surge
                } else {
                    0.0
                };

                (target_eta, normal_velocity)
            }
            EdgeBoundary::ConstantInflow { target_depth, inflow_velocity } => {
                (target_depth, inflow_velocity)
            }
            _ => (0.0, 0.0),
        }
    }
}

/// Specification of boundary conditions for all four domain perimeters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainBoundaryConfig {
    pub north: EdgeBoundary,
    pub south: EdgeBoundary,
    pub east: EdgeBoundary,
    pub west: EdgeBoundary,
}

impl Default for DomainBoundaryConfig {
    fn default() -> Self {
        Self {
            north: EdgeBoundary::SolidWall,
            south: EdgeBoundary::SolidWall,
            east: EdgeBoundary::SolidWall,
            west: EdgeBoundary::SolidWall,
        }
    }
}

impl DomainBoundaryConfig {
    /// Standard closed basin with solid reflective walls on all edges.
    pub fn all_solid() -> Self {
        Self::default()
    }

    /// Coastal beach scenario: wave generator on the South (ocean) edge,
    /// solid reflective walls on North/East/West flanks so waves surge forward without leaking sideways.
    pub fn coastal_waves(wave_gen: EdgeBoundary) -> Self {
        Self {
            north: EdgeBoundary::SolidWall,
            south: wave_gen,
            east: EdgeBoundary::SolidWall,
            west: EdgeBoundary::SolidWall,
        }
    }

    /// River scenario: continuous inflow at North, ocean outflow sink at South.
    pub fn river_inflow_outflow(inflow: EdgeBoundary, outflow: EdgeBoundary) -> Self {
        Self {
            north: inflow,
            south: outflow,
            east: EdgeBoundary::SolidWall,
            west: EdgeBoundary::SolidWall,
        }
    }
}

/// Applies domain boundary conditions to the 1-cell ghost halo ring of a GridState.
pub fn apply_domain_boundaries(
    state: &mut GridState,
    boundaries: &DomainBoundaryConfig,
    time: f32,
) {
    let width = state.width;
    let height = state.height;

    if width < 3 || height < 3 {
        return;
    }

    // --- 1. North Edge (y = 0, ghost neighbor is y = 1) ---
    apply_edge_h(state, 0, 1, width, boundaries.north, time, true);

    // --- 2. South Edge (y = height - 1, ghost neighbor is y = height - 2) ---
    apply_edge_h(state, height - 1, height - 2, width, boundaries.south, time, false);

    // --- 3. West Edge (x = 0, ghost neighbor is x = 1) ---
    apply_edge_v(state, 0, 1, height, boundaries.west, time, true);

    // --- 4. East Edge (x = width - 1, ghost neighbor is x = width - 2) ---
    apply_edge_v(state, width - 1, width - 2, height, boundaries.east, time, false);

    // --- 5. Four Corners (Diagonal reflection) ---
    let corners = [
        (0, 0, 1, 1),
        (width - 1, 0, width - 2, 1),
        (0, height - 1, 1, height - 2),
        (width - 1, height - 1, width - 2, height - 2),
    ];
    for (gx, gy, ix, iy) in corners {
        let g_idx = state.idx(gx, gy);
        let i_idx = state.idx(ix, iy);
        state.h[g_idx] = state.h[i_idx];
        state.z_bed[g_idx] = state.z_bed[i_idx];
        state.u[g_idx] = -state.u[i_idx];
        state.v[g_idx] = -state.v[i_idx];
        state.sediment_c[g_idx] = state.sediment_c[i_idx];
        state.soil_sat[g_idx] = state.soil_sat[i_idx];
        state.bedrock_z[g_idx] = state.bedrock_z[i_idx];
    }
}

/// Applies boundary condition along a horizontal row (North or South).
fn apply_edge_h(
    state: &mut GridState,
    y_ghost: u32,
    y_interior: u32,
    width: u32,
    boundary: EdgeBoundary,
    time: f32,
    is_north: bool,
) {
    for x in 1..(width - 1) {
        let g_idx = state.idx(x, y_ghost);
        let i_idx = state.idx(x, y_interior);
        let z_bed = state.z_bed[i_idx];
        state.z_bed[g_idx] = z_bed;
        state.bedrock_z[g_idx] = state.bedrock_z[i_idx];
        state.soil_sat[g_idx] = state.soil_sat[i_idx];
        state.sediment_c[g_idx] = state.sediment_c[i_idx];

        match boundary {
            EdgeBoundary::SolidWall => {
                state.h[g_idx] = state.h[i_idx];
                state.u[g_idx] = state.u[i_idx];
                state.v[g_idx] = -state.v[i_idx]; // Reflect normal velocity
            }
            EdgeBoundary::OpenOutflow { absorption_rate } => {
                let factor = (1.0 - absorption_rate).clamp(0.0, 1.0);
                state.h[g_idx] = state.h[i_idx] * factor;
                state.u[g_idx] = state.u[i_idx] * factor;
                // Allow water exiting the boundary, suppress incoming reflection
                let v_in = state.v[i_idx];
                state.v[g_idx] = if is_north {
                    if v_in < 0.0 { v_in * factor } else { 0.0 }
                } else {
                    if v_in > 0.0 { v_in * factor } else { 0.0 }
                };
            }
            EdgeBoundary::ConstantInflow { target_depth, inflow_velocity } => {
                state.h[g_idx] = target_depth;
                state.u[g_idx] = 0.0;
                state.v[g_idx] = if is_north { inflow_velocity } else { -inflow_velocity };
            }
            EdgeBoundary::WaveGenerator { .. } => {
                let (target_eta, surge_speed) = boundary.evaluate_wave(time);
                let target_h = (target_eta - z_bed).max(0.0);

                if surge_speed > 0.01 {
                    // Surge phase: push water into domain with onshore velocity
                    state.h[g_idx] = target_h;
                    state.u[g_idx] = 0.0;
                    state.v[g_idx] = if is_north { surge_speed } else { -surge_speed };
                } else {
                    // Backwash phase: water drains out into ocean trough
                    let v_in = state.v[i_idx];
                    let outgoing = if is_north { v_in < 0.0 } else { v_in > 0.0 };
                    state.h[g_idx] = state.h[i_idx].min(target_h);
                    state.u[g_idx] = state.u[i_idx] * 0.8;
                    state.v[g_idx] = if outgoing { v_in } else { 0.0 };
                }
            }
        }
    }
}

/// Applies boundary condition along a vertical column (West or East).
fn apply_edge_v(
    state: &mut GridState,
    x_ghost: u32,
    x_interior: u32,
    height: u32,
    boundary: EdgeBoundary,
    time: f32,
    is_west: bool,
) {
    for y in 1..(height - 1) {
        let g_idx = state.idx(x_ghost, y);
        let i_idx = state.idx(x_interior, y);
        let z_bed = state.z_bed[i_idx];
        state.z_bed[g_idx] = z_bed;
        state.bedrock_z[g_idx] = state.bedrock_z[i_idx];
        state.soil_sat[g_idx] = state.soil_sat[i_idx];
        state.sediment_c[g_idx] = state.sediment_c[i_idx];

        match boundary {
            EdgeBoundary::SolidWall => {
                state.h[g_idx] = state.h[i_idx];
                state.u[g_idx] = -state.u[i_idx]; // Reflect normal velocity
                state.v[g_idx] = state.v[i_idx];
            }
            EdgeBoundary::OpenOutflow { absorption_rate } => {
                let factor = (1.0 - absorption_rate).clamp(0.0, 1.0);
                state.h[g_idx] = state.h[i_idx] * factor;
                state.v[g_idx] = state.v[i_idx] * factor;
                let u_in = state.u[i_idx];
                state.u[g_idx] = if is_west {
                    if u_in < 0.0 { u_in * factor } else { 0.0 }
                } else {
                    if u_in > 0.0 { u_in * factor } else { 0.0 }
                };
            }
            EdgeBoundary::ConstantInflow { target_depth, inflow_velocity } => {
                state.h[g_idx] = target_depth;
                state.u[g_idx] = if is_west { inflow_velocity } else { -inflow_velocity };
                state.v[g_idx] = 0.0;
            }
            EdgeBoundary::WaveGenerator { .. } => {
                let (target_eta, surge_speed) = boundary.evaluate_wave(time);
                let target_h = (target_eta - z_bed).max(0.0);

                if surge_speed > 0.01 {
                    state.h[g_idx] = target_h;
                    state.u[g_idx] = if is_west { surge_speed } else { -surge_speed };
                    state.v[g_idx] = 0.0;
                } else {
                    let u_in = state.u[i_idx];
                    let outgoing = if is_west { u_in < 0.0 } else { u_in > 0.0 };
                    state.h[g_idx] = state.h[i_idx].min(target_h);
                    state.u[g_idx] = if outgoing { u_in } else { 0.0 };
                    state.v[g_idx] = state.v[i_idx] * 0.8;
                }
            }
        }
    }
}
