use crate::state::DoubleBufferedGrid;
use crate::state::GridState;

pub const DEFAULT_GRAVITY: f32 = 9.81;

/// Parameters governing Shallow Water Equations physics.
#[derive(Copy, Clone, Debug)]
pub struct SweParams {
    /// Manning roughness coefficient 'n' (SI: s / m^(1/3))
    pub manning_n: f32,
    /// Threshold depth (meters) below which a cell is treated as dry
    pub h_dry: f32,
    /// Gravitational acceleration (m / s^2)
    pub gravity: f32,
}

impl Default for SweParams {
    fn default() -> Self {
        Self {
            manning_n: 0.025,
            h_dry: 1e-4,
            gravity: DEFAULT_GRAVITY,
        }
    }
}

/// Computes the maximum stable time step according to the Courant-Friedrichs-Lewy (CFL) condition:
///
/// dt <= CFL * min(dx, dy) / max(|v| + sqrt(g * h))
pub fn compute_max_stable_dt(grid: &GridState, dx: f32, dy: f32, cfl: f32, g: f32) -> f32 {
    let mut max_speed = 1e-4f32;

    for y in 1..(grid.height - 1) {
        for x in 1..(grid.width - 1) {
            let idx = grid.idx(x, y);
            let h = grid.h[idx];
            if h > 1e-4 {
                let u = grid.u[idx];
                let v = grid.v[idx];
                let speed = (u * u + v * v).sqrt() + (g * h).sqrt();
                if speed > max_speed {
                    max_speed = speed;
                }
            }
        }
    }

    let min_delta = dx.min(dy);
    cfl * min_delta / max_speed
}

/// Performs a single simulation step of the Shallow Water Equations using default parameters.
pub fn step_swe(grid: &mut DoubleBufferedGrid, dt: f32) {
    step_swe_with_params(grid, dt, &SweParams::default());
}

/// Performs a single simulation step of the Shallow Water Equations with custom parameters.
///
/// Implements a well-balanced finite volume scheme with:
/// - Audusse et al. (2004) hydrostatic reconstruction for the well-balanced "lake at rest" property
/// - Rusanov (Local Lax-Friedrichs) numerical interface fluxes
/// - Strict zero-normal-flux reflective ghost halo boundary conditions
/// - Non-destructive positivity-preserving depth updates
/// - Semi-implicit Manning friction drag
pub fn step_swe_with_params(grid: &mut DoubleBufferedGrid, dt: f32, params: &SweParams) {
    let dx = grid.descriptor.extent_x / grid.descriptor.grid_res_x as f32;
    let dy = grid.descriptor.extent_y / grid.descriptor.grid_res_y as f32;
    let width = grid.descriptor.grid_res_x;
    let height = grid.descriptor.grid_res_y;
    let g = params.gravity;
    let h_dry = params.h_dry;
    let manning_n = params.manning_n;

    // Ensure ghost cells on the current buffer are up to date before computing interface fluxes
    grid.current.apply_reflective_boundaries();

    // Helper for local wave speed with protection against negative depths
    let wave_speed = |h: f32, v: f32| -> f32 {
        v.abs() + (g * h.max(0.0)).sqrt()
    };

    // Flux function for Shallow Water Equations:
    // F(U) = [h*u, h*u^2 + 0.5*g*h^2, h*u*v]
    let calc_flux = |h: f32, u_normal: f32, v_tangent: f32| -> (f32, f32, f32) {
        (
            h * u_normal,
            h * u_normal * u_normal + 0.5 * g * h * h,
            h * u_normal * v_tangent,
        )
    };

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = grid.current.idx(x, y);

            let h_c = grid.current.h[idx];
            let u_c = grid.current.u[idx];
            let v_c = grid.current.v[idx];
            let z_c = grid.current.z_bed[idx];

            let idx_l = grid.current.idx(x - 1, y);
            let idx_r = grid.current.idx(x + 1, y);
            let idx_t = grid.current.idx(x, y - 1);
            let idx_b = grid.current.idx(x, y + 1);

            let h_l = grid.current.h[idx_l];
            let h_r = grid.current.h[idx_r];
            let h_t = grid.current.h[idx_t];
            let h_b = grid.current.h[idx_b];

            // Fast path: if cell and all 4 neighbors are dry, fluid remains at rest
            if h_c <= h_dry && h_l <= h_dry && h_r <= h_dry && h_t <= h_dry && h_b <= h_dry {
                grid.next.h[idx] = h_c.max(0.0);
                grid.next.u[idx] = 0.0;
                grid.next.v[idx] = 0.0;
                grid.next.z_bed[idx] = z_c;
                grid.next.sediment_c[idx] = grid.current.sediment_c[idx];
                grid.next.soil_sat[idx] = grid.current.soil_sat[idx];
                grid.next.bedrock_z[idx] = grid.current.bedrock_z[idx];
                continue;
            }

            let u_l = grid.current.u[idx_l];
            let u_r = grid.current.u[idx_r];
            let v_t = grid.current.v[idx_t];
            let v_b = grid.current.v[idx_b];

            let z_r = grid.current.z_bed[idx_r];
            let z_l = grid.current.z_bed[idx_l];
            let z_b = grid.current.z_bed[idx_b];
            let z_t = grid.current.z_bed[idx_t];

            // --- Hydrostatic Reconstruction (Audusse et al. 2004) ---
            // Reconstruct water depths at cell interfaces to guarantee the Well-Balanced property

            // X-interfaces
            let z_max_l = z_l.max(z_c);
            let h_l_xl = (h_l + z_l - z_max_l).max(0.0);
            let h_r_xl = (h_c + z_c - z_max_l).max(0.0);

            let z_max_r = z_c.max(z_r);
            let h_l_xr = (h_c + z_c - z_max_r).max(0.0);
            let h_r_xr = (h_r + z_r - z_max_r).max(0.0);

            // Y-interfaces
            let z_max_t = z_t.max(z_c);
            let h_l_yt = (h_t + z_t - z_max_t).max(0.0);
            let h_r_yt = (h_c + z_c - z_max_t).max(0.0);

            let z_max_b = z_c.max(z_b);
            let h_l_yb = (h_c + z_c - z_max_b).max(0.0);
            let h_r_yb = (h_b + z_b - z_max_b).max(0.0);

            // --- X Fluxes (F) ---
            let f_l_xl = calc_flux(h_l_xl, u_l, grid.current.v[idx_l]);
            let f_r_xl = calc_flux(h_r_xl, u_c, v_c);

            let f_l_xr = calc_flux(h_l_xr, u_c, v_c);
            let f_r_xr = calc_flux(h_r_xr, u_r, grid.current.v[idx_r]);

            let a_x_l = wave_speed(h_l_xl, u_l).max(wave_speed(h_r_xl, u_c));
            let a_x_r = wave_speed(h_l_xr, u_c).max(wave_speed(h_r_xr, u_r));

            let flux_x_left_h = 0.5 * (f_l_xl.0 + f_r_xl.0) - 0.5 * a_x_l * (h_r_xl - h_l_xl);
            let flux_x_left_hu = 0.5 * (f_l_xl.1 + f_r_xl.1) - 0.5 * a_x_l * (h_r_xl * u_c - h_l_xl * u_l);
            let flux_x_left_hv = 0.5 * (f_l_xl.2 + f_r_xl.2) - 0.5 * a_x_l * (h_r_xl * v_c - h_l_xl * grid.current.v[idx_l]);

            let flux_x_right_h = 0.5 * (f_l_xr.0 + f_r_xr.0) - 0.5 * a_x_r * (h_r_xr - h_l_xr);
            let flux_x_right_hu = 0.5 * (f_l_xr.1 + f_r_xr.1) - 0.5 * a_x_r * (h_r_xr * u_r - h_l_xr * u_c);
            let flux_x_right_hv = 0.5 * (f_l_xr.2 + f_r_xr.2) - 0.5 * a_x_r * (h_r_xr * grid.current.v[idx_r] - h_l_xr * v_c);

            // --- Y Fluxes (G) ---
            let g_l_yt = calc_flux(h_l_yt, v_t, grid.current.u[idx_t]);
            let g_r_yt = calc_flux(h_r_yt, v_c, u_c);

            let g_l_yb = calc_flux(h_l_yb, v_c, u_c);
            let g_r_yb = calc_flux(h_r_yb, v_b, grid.current.u[idx_b]);

            let a_y_t = wave_speed(h_l_yt, v_t).max(wave_speed(h_r_yt, v_c));
            let a_y_b = wave_speed(h_l_yb, v_c).max(wave_speed(h_r_yb, v_b));

            let flux_y_top_h = 0.5 * (g_l_yt.0 + g_r_yt.0) - 0.5 * a_y_t * (h_r_yt - h_l_yt);
            let flux_y_top_hu = 0.5 * (g_l_yt.2 + g_r_yt.2) - 0.5 * a_y_t * (h_r_yt * u_c - h_l_yt * grid.current.u[idx_t]);
            let flux_y_top_hv = 0.5 * (g_l_yt.1 + g_r_yt.1) - 0.5 * a_y_t * (h_r_yt * v_c - h_l_yt * v_t);

            let flux_y_bottom_h = 0.5 * (g_l_yb.0 + g_r_yb.0) - 0.5 * a_y_b * (h_r_yb - h_l_yb);
            let flux_y_bottom_hu = 0.5 * (g_l_yb.2 + g_r_yb.2) - 0.5 * a_y_b * (h_r_yb * grid.current.u[idx_b] - h_l_yb * u_c);
            let flux_y_bottom_hv = 0.5 * (g_l_yb.1 + g_r_yb.1) - 0.5 * a_y_b * (h_r_yb * v_b - h_l_yb * v_c);

            // --- Well-Balanced Bed Slope Source Terms ---
            let source_hu = 0.5 * g * (h_l_xr * h_l_xr - h_r_xl * h_r_xl) / dx;
            let source_hv = 0.5 * g * (h_l_yb * h_l_yb - h_r_yt * h_r_yt) / dy;

            // --- State Update ---
            let dh_dt = -((flux_x_right_h - flux_x_left_h) / dx + (flux_y_bottom_h - flux_y_top_h) / dy);
            let dhu_dt = -((flux_x_right_hu - flux_x_left_hu) / dx + (flux_y_bottom_hu - flux_y_top_hu) / dy) + source_hu;
            let dhv_dt = -((flux_x_right_hv - flux_x_left_hv) / dx + (flux_y_bottom_hv - flux_y_top_hv) / dy) + source_hv;

            // Conservative Suspended Sediment Advection (Flux Consistent with SWE)
            let c_c = grid.current.sediment_c[idx];
            let c_l = grid.current.sediment_c[idx_l];
            let c_r = grid.current.sediment_c[idx_r];
            let c_t = grid.current.sediment_c[idx_t];
            let c_b = grid.current.sediment_c[idx_b];

            let flux_x_left_s = if flux_x_left_h > 0.0 { flux_x_left_h * c_l } else { flux_x_left_h * c_c };
            let flux_x_right_s = if flux_x_right_h > 0.0 { flux_x_right_h * c_c } else { flux_x_right_h * c_r };
            let flux_y_top_s = if flux_y_top_h > 0.0 { flux_y_top_h * c_t } else { flux_y_top_h * c_c };
            let flux_y_bottom_s = if flux_y_bottom_h > 0.0 { flux_y_bottom_h * c_c } else { flux_y_bottom_h * c_b };

            let dqs_dt = -((flux_x_right_s - flux_x_left_s) / dx + (flux_y_bottom_s - flux_y_top_s) / dy);
            let qs_c = h_c * c_c;
            let qs_next = (qs_c + dqs_dt * dt).max(0.0);

            let mut h_next = h_c + dh_dt * dt;
            let hu_next = (h_c * u_c) + dhu_dt * dt;
            let hv_next = (h_c * v_c) + dhv_dt * dt;

            // Positivity-preserving depth update and velocity recovery
            let u_next;
            let v_next;
            let mut z_next = z_c;
            let c_next;

            if h_next <= h_dry {
                // Keep the small water depth (clamping only negligible numerical undershoot)
                // without deleting mass or creating velocities
                h_next = h_next.max(0.0);
                u_next = 0.0;
                v_next = 0.0;
                // If cell dries out, any remaining suspended sediment settles to bed (porosity p = 0.40)
                let inv_one_minus_p = 1.0 / (1.0 - 0.40);
                z_next += qs_next * inv_one_minus_p;
                c_next = 0.0;
            } else {
                c_next = (qs_next / h_next).clamp(0.0, 0.50);

                // Desingularized velocity recovery: u = h * (hu) / (h^2 + h_dry^2)
                // Prevents artificial velocity singularities at thin wetting fronts
                let denom = h_next * h_next + h_dry * h_dry;
                let mut raw_u = (h_next * hu_next) / denom;
                let mut raw_v = (h_next * hv_next) / denom;

                // Physical velocity ceiling (Froude limiter) to prevent numerical blowup
                let raw_speed = (raw_u * raw_u + raw_v * raw_v).sqrt();
                let max_speed = 20.0f32;
                if raw_speed > max_speed {
                    let scale = max_speed / raw_speed;
                    raw_u *= scale;
                    raw_v *= scale;
                }

                // Semi-implicit Manning friction drag:
                // S_f = g * n^2 * |v| * v / h^(4/3)
                if manning_n > 0.0 {
                    let speed = (raw_u * raw_u + raw_v * raw_v).sqrt();
                    let h_eff = h_next.max(h_dry);
                    let drag = dt * g * manning_n * manning_n * speed / h_eff.powf(4.0 / 3.0);
                    let friction_factor = 1.0 / (1.0 + drag);
                    u_next = raw_u * friction_factor;
                    v_next = raw_v * friction_factor;
                } else {
                    u_next = raw_u;
                    v_next = raw_v;
                }
            }

            grid.next.h[idx] = h_next;
            grid.next.u[idx] = u_next;
            grid.next.v[idx] = v_next;
            grid.next.z_bed[idx] = z_next;
            grid.next.sediment_c[idx] = c_next;
            grid.next.soil_sat[idx] = grid.current.soil_sat[idx];
            grid.next.bedrock_z[idx] = grid.current.bedrock_z[idx];
        }
    }

    // Apply reflective boundary conditions to the new state
    grid.next.apply_reflective_boundaries();

    // Swap buffers for next tick
    grid.swap();
}
