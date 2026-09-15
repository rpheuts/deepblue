struct SimDomain {
    extent_x: f32,
    extent_y: f32,
    max_elevation: f32,
    grid_res_x: u32,
    grid_res_y: u32,
    world_origin_x: f32,
    world_origin_y: f32,
    world_origin_z: f32,
}

struct StepParams {
    dt: f32,
    time: f32,
    south_type: u32,
    south_base_eta: f32,

    south_wave_amp: f32,
    south_wave_period: f32,
    south_surge_speed: f32,
    south_tide_amp: f32,

    south_tide_period: f32,
    south_outflow_rate: f32,
    north_type: u32,
    north_inflow_h: f32,

    north_inflow_v: f32,
    north_outflow_rate: f32,
    west_type: u32,
    east_type: u32,

    west_outflow_rate: f32,
    east_outflow_rate: f32,
    stream_inflow_active: f32,
    coastal_sink_active: f32,

    wind_speed: f32,
    wind_dir_x: f32,
    wind_dir_y: f32,
    wind_drag_coeff: f32,

    wind_turbulence: f32,
    wind_shelter: f32,
    step_param_pad1: f32,
    step_param_pad2: f32,
}

@group(0) @binding(0) var<uniform> domain: SimDomain;
@group(0) @binding(1) var<uniform> params: StepParams;

// In buffers (read-only for current time step)
@group(0) @binding(2) var<storage, read> in_h: array<f32>;
@group(0) @binding(3) var<storage, read> in_u: array<f32>;
@group(0) @binding(4) var<storage, read> in_v: array<f32>;
@group(0) @binding(5) var<storage, read> in_z: array<f32>;

// Out buffers (written for next time step)
@group(0) @binding(6) var<storage, read_write> out_h: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_u: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_v: array<f32>;
@group(0) @binding(9) var<storage, read_write> out_z: array<f32>;

const G: f32 = 9.81;

fn get_idx(x: u32, y: u32) -> u32 {
    return y * domain.grid_res_x + x;
}

fn wave_speed(h: f32, v: f32) -> f32 {
    return abs(v) + sqrt(G * max(0.0, h));
}

fn calc_f(h: f32, u: f32, v: f32) -> vec3<f32> {
    return vec3<f32>(h * u, h * u * u + 0.5 * G * h * h, h * u * v);
}

fn calc_g(h: f32, u: f32, v: f32) -> vec3<f32> {
    return vec3<f32>(h * v, h * u * v, h * v * v + 0.5 * G * h * h);
}

fn calc_wind_turbulence(world_pos: vec2<f32>, time: f32, mean_dir: vec2<f32>, mean_speed: f32, turb_scale: f32) -> vec2<f32> {
    if (turb_scale <= 0.001 || mean_speed <= 0.01) {
        return mean_dir * mean_speed;
    }
    // Taylor's frozen turbulence hypothesis: advect eddies downwind at wind speed
    let adv_pos = world_pos - mean_dir * (mean_speed * time * 0.65);

    // Large-scale atmospheric gust intermittency envelope G(x, t) in [0.35, 1.50]
    // Creates moving gust pockets ("cat's paws") separated by calm lulls
    let env_arg1 = 0.045 * adv_pos.x + 0.038 * adv_pos.y;
    let env_arg2 = 0.041 * adv_pos.y - 0.032 * adv_pos.x + 0.7;
    let gust_raw = 0.5 + 0.5 * sin(env_arg1) * cos(env_arg2);
    let gust_envelope = 0.35 + 1.15 * smoothstep(0.25, 0.75, gust_raw);

    // Multi-rotor golden-ratio coordinate rotations: prevents repeating lattice patterns
    // Rotor 1: theta1 = 0.65 rad (cos = 0.796, sin = 0.605)
    let p1 = vec2<f32>(0.796 * adv_pos.x + 0.605 * adv_pos.y, -0.605 * adv_pos.x + 0.796 * adv_pos.y);
    // Rotor 2: theta2 = 1.83 rad (cos = -0.255, sin = 0.967)
    let p2 = vec2<f32>(-0.255 * adv_pos.x + 0.967 * adv_pos.y, -0.967 * adv_pos.x - 0.255 * adv_pos.y);
    // Rotor 3: theta3 = 2.91 rad (cos = -0.974, sin = 0.228)
    let p3 = vec2<f32>(-0.974 * adv_pos.x + 0.228 * adv_pos.y, -0.228 * adv_pos.x - 0.974 * adv_pos.y);

    // Incommensurate frequencies: f1 = 0.11, f2 = 0.27, f3 = 0.68
    let u1 = 0.11 * p1.x + cos(0.09 * p1.y);
    let u2 = 0.27 * p2.x - 0.23 * p2.y + 1.4;
    let u3 = 0.68 * p3.x + 0.54 * p3.y - 0.8;

    // Derivatives for Rotor 1:
    let du1_dp1x = 0.11;
    let du1_dp1y = -0.09 * sin(0.09 * p1.y);
    let du1_dx = du1_dp1x * 0.796 - du1_dp1y * 0.605;
    let du1_dy = du1_dp1x * 0.605 + du1_dp1y * 0.796;

    // Derivatives for Rotor 2:
    let du2_dx = -0.2913;
    let du2_dy = 0.2024;

    // Derivatives for Rotor 3:
    let du3_dx = -0.7854;
    let du3_dy = -0.3709;

    let dpsi_dx = cos(u1) * du1_dx + 0.55 * cos(u2) * du2_dx - 0.28 * sin(u3) * du3_dx;
    let dpsi_dy = cos(u1) * du1_dy + 0.55 * cos(u2) * du2_dy - 0.28 * sin(u3) * du3_dy;

    // Divergence-free curl noise: w' = (dpsi/dy, -dpsi/dx)
    let curl_vec = vec2<f32>(dpsi_dy, -dpsi_dx);

    // Modulate by the intermittent gust envelope
    let effective_turb = turb_scale * gust_envelope * mean_speed * 0.75;
    return mean_dir * mean_speed + curl_vec * effective_turb;
}

fn calc_orographic_shelter_swe(gx: u32, gy: u32, z_c: f32, dx: f32, turb_wind: vec2<f32>, shelter_strength: f32) -> f32 {
    if (shelter_strength <= 0.001) {
        return 1.0;
    }
    let turb_len = length(turb_wind);
    if (turb_len <= 0.01) {
        return 1.0;
    }
    // Look upwind along the local turbulent wind vector (dynamically meanders with gusts)
    let upwind_dir = -turb_wind / turb_len;
    let cell_size = max(dx, 0.01);

    // 3-Ray Angular Wake Fan (+-15 degrees lateral spreading for conical wake)
    let dir_c = upwind_dir;
    let dir_l = vec2<f32>(upwind_dir.x * 0.966 - upwind_dir.y * 0.259, upwind_dir.x * 0.259 + upwind_dir.y * 0.966);
    let dir_r = vec2<f32>(upwind_dir.x * 0.966 + upwind_dir.y * 0.259, -upwind_dir.x * 0.259 + upwind_dir.y * 0.966);

    let max_gx = i32(domain.grid_res_x - 1u);
    let max_gy = i32(domain.grid_res_y - 1u);
    let pos = vec2<f32>(f32(gx), f32(gy));

    // Sample distances: near (1.8m), mid (4.0m), far (7.5m)
    let d1 = 1.8 / cell_size;
    let d2 = 4.0 / cell_size;
    let d3 = 7.5 / cell_size;

    // Center ray sampling
    let c1 = clamp(i32(round(pos.x + dir_c.x * d1)), 0, max_gx);
    let c1y = clamp(i32(round(pos.y + dir_c.y * d1)), 0, max_gy);
    let c2 = clamp(i32(round(pos.x + dir_c.x * d2)), 0, max_gx);
    let c2y = clamp(i32(round(pos.y + dir_c.y * d2)), 0, max_gy);
    let c3 = clamp(i32(round(pos.x + dir_c.x * d3)), 0, max_gx);
    let c3y = clamp(i32(round(pos.y + dir_c.y * d3)), 0, max_gy);
    let z_c1 = in_z[u32(c1y) * domain.grid_res_x + u32(c1)];
    let z_c2 = in_z[u32(c2y) * domain.grid_res_x + u32(c2)];
    let z_c3 = in_z[u32(c3y) * domain.grid_res_x + u32(c3)];
    let dz_c = max(0.0, max(z_c1, max(z_c2, z_c3)) - z_c);

    // Left flank ray sampling
    let l1 = clamp(i32(round(pos.x + dir_l.x * d1)), 0, max_gx);
    let l1y = clamp(i32(round(pos.y + dir_l.y * d1)), 0, max_gy);
    let l2 = clamp(i32(round(pos.x + dir_l.x * d2)), 0, max_gx);
    let l2y = clamp(i32(round(pos.y + dir_l.y * d2)), 0, max_gy);
    let l3 = clamp(i32(round(pos.x + dir_l.x * d3)), 0, max_gx);
    let l3y = clamp(i32(round(pos.y + dir_l.y * d3)), 0, max_gy);
    let z_l1 = in_z[u32(l1y) * domain.grid_res_x + u32(l1)];
    let z_l2 = in_z[u32(l2y) * domain.grid_res_x + u32(l2)];
    let z_l3 = in_z[u32(l3y) * domain.grid_res_x + u32(l3)];
    let dz_l = max(0.0, max(z_l1, max(z_l2, z_l3)) - z_c);

    // Right flank ray sampling
    let r1 = clamp(i32(round(pos.x + dir_r.x * d1)), 0, max_gx);
    let r1y = clamp(i32(round(pos.y + dir_r.y * d1)), 0, max_gy);
    let r2 = clamp(i32(round(pos.x + dir_r.x * d2)), 0, max_gx);
    let r2y = clamp(i32(round(pos.y + dir_r.y * d2)), 0, max_gy);
    let r3 = clamp(i32(round(pos.x + dir_r.x * d3)), 0, max_gx);
    let r3y = clamp(i32(round(pos.y + dir_r.y * d3)), 0, max_gy);
    let z_r1 = in_z[u32(r1y) * domain.grid_res_x + u32(r1)];
    let z_r2 = in_z[u32(r2y) * domain.grid_res_x + u32(r2)];
    let z_r3 = in_z[u32(r3y) * domain.grid_res_x + u32(r3)];
    let dz_r = max(0.0, max(z_r1, max(z_r2, z_r3)) - z_c);

    // Conical weighted integration: center = 50%, flanks = 25% each
    let effective_dz = 0.50 * dz_c + 0.25 * dz_l + 0.25 * dz_r;

    // Subtle edge feathering to soften boundary
    let edge_jitter = 0.04 * sin(0.85 * f32(gx) + 0.72 * f32(gy));
    let shelter_val = exp(-2.0 * shelter_strength * max(0.0, effective_dz + edge_jitter));
    return clamp(shelter_val, 0.02, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = domain.grid_res_x;
    let height = domain.grid_res_y;
    let dt = params.dt;

    // Interior domain only (ghost halo handled in boundary pass)
    if (x == 0u || x >= width - 1u || y == 0u || y >= height - 1u) {
        return;
    }

    let dx = domain.extent_x / f32(width);
    let dy = domain.extent_y / f32(height);

    let idx = get_idx(x, y);
    let idx_l = get_idx(x - 1u, y);
    let idx_r = get_idx(x + 1u, y);
    let idx_t = get_idx(x, y - 1u);
    let idx_b = get_idx(x, y + 1u);

    // Fetch primitives
    let h_c = in_h[idx]; let u_c = in_u[idx]; let v_c = in_v[idx]; let z_c = in_z[idx];
    let h_l = in_h[idx_l]; let u_l = in_u[idx_l]; let v_l = in_v[idx_l]; let z_l = in_z[idx_l];
    let h_r = in_h[idx_r]; let u_r = in_u[idx_r]; let v_r = in_v[idx_r]; let z_r = in_z[idx_r];
    let h_t = in_h[idx_t]; let u_t = in_u[idx_t]; let v_t = in_v[idx_t]; let z_t = in_z[idx_t];
    let h_b = in_h[idx_b]; let u_b = in_u[idx_b]; let v_b = in_v[idx_b]; let z_b = in_z[idx_b];

    let h_dry = 1e-4;

    // Fast path: dry cells remain at rest
    if (h_c <= h_dry && h_l <= h_dry && h_r <= h_dry && h_t <= h_dry && h_b <= h_dry) {
        out_h[idx] = max(0.0, h_c);
        out_u[idx] = 0.0;
        out_v[idx] = 0.0;
        out_z[idx] = z_c;
        return;
    }

    // --- Hydrostatic Reconstruction (Audusse et al.) ---
    let z_max_l = max(z_l, z_c);
    let h_l_xl = max(0.0, h_l + z_l - z_max_l);
    let h_r_xl = max(0.0, h_c + z_c - z_max_l);

    let z_max_r = max(z_c, z_r);
    let h_l_xr = max(0.0, h_c + z_c - z_max_r);
    let h_r_xr = max(0.0, h_r + z_r - z_max_r);

    let z_max_t = max(z_t, z_c);
    let h_l_yt = max(0.0, h_t + z_t - z_max_t);
    let h_r_yt = max(0.0, h_c + z_c - z_max_t);

    let z_max_b = max(z_c, z_b);
    let h_l_yb = max(0.0, h_c + z_c - z_max_b);
    let h_r_yb = max(0.0, h_b + z_b - z_max_b);

    // --- X Fluxes ---
    let f_l_xl = calc_f(h_l_xl, u_l, v_l);
    let f_r_xl = calc_f(h_r_xl, u_c, v_c);
    let f_l_xr = calc_f(h_l_xr, u_c, v_c);
    let f_r_xr = calc_f(h_r_xr, u_r, v_r);

    let a_x_l = max(wave_speed(h_l_xl, u_l), wave_speed(h_r_xl, u_c));
    let a_x_r = max(wave_speed(h_l_xr, u_c), wave_speed(h_r_xr, u_r));

    let flux_x_left = 0.5 * (f_l_xl + f_r_xl) - 0.5 * a_x_l * vec3<f32>(
        h_r_xl - h_l_xl,
        h_r_xl * u_c - h_l_xl * u_l,
        h_r_xl * v_c - h_l_xl * v_l
    );

    let flux_x_right = 0.5 * (f_l_xr + f_r_xr) - 0.5 * a_x_r * vec3<f32>(
        h_r_xr - h_l_xr,
        h_r_xr * u_r - h_l_xr * u_c,
        h_r_xr * v_r - h_l_xr * v_c
    );

    // --- Y Fluxes ---
    let g_l_yt = calc_g(h_l_yt, u_t, v_t);
    let g_r_yt = calc_g(h_r_yt, u_c, v_c);
    let g_l_yb = calc_g(h_l_yb, u_c, v_c);
    let g_r_yb = calc_g(h_r_yb, u_b, v_b);

    let a_y_t = max(wave_speed(h_l_yt, v_t), wave_speed(h_r_yt, v_c));
    let a_y_b = max(wave_speed(h_l_yb, v_c), wave_speed(h_r_yb, v_b));

    let flux_y_top = 0.5 * (g_l_yt + g_r_yt) - 0.5 * a_y_t * vec3<f32>(
        h_r_yt - h_l_yt,
        h_r_yt * u_c - h_l_yt * u_t,
        h_r_yt * v_c - h_l_yt * v_t
    );

    let flux_y_bottom = 0.5 * (g_l_yb + g_r_yb) - 0.5 * a_y_b * vec3<f32>(
        h_r_yb - h_l_yb,
        h_r_yb * u_b - h_l_yb * u_c,
        h_r_yb * v_b - h_l_yb * v_c
    );

    // --- Source terms (Bed slope) ---
    let source_hu = 0.5 * G * (h_l_xr * h_l_xr - h_r_xl * h_r_xl) / dx;
    let source_hv = 0.5 * G * (h_l_yb * h_l_yb - h_r_yt * h_r_yt) / dy;

    // --- State Update ---
    let dq_dt = -((flux_x_right - flux_x_left) / dx + (flux_y_bottom - flux_y_top) / dy);
    let dq_dt_total = dq_dt + vec3<f32>(0.0, source_hu, source_hv);

    var h_next = h_c + dq_dt_total.x * dt;
    let hu_next = (h_c * u_c) + dq_dt_total.y * dt;
    let hv_next = (h_c * v_c) + dq_dt_total.z * dt;

    var u_next = 0.0;
    var v_next = 0.0;

    if (h_next <= h_dry) {
        h_next = max(0.0, h_next);
        u_next = 0.0;
        v_next = 0.0;
    } else {
        // Desingularized velocity recovery: u = h * (hu) / (h^2 + h_dry^2)
        let denom = h_next * h_next + h_dry * h_dry;
        var raw_u = (h_next * hu_next) / denom;
        var raw_v = (h_next * hv_next) / denom;

        // Aerodynamic wind surface shear stress with divergence-free turbulence & orographic sheltering
        let mean_wind_speed = params.wind_speed;
        if (mean_wind_speed > 0.01 && h_next > 0.01) {
            let mean_dir = vec2<f32>(params.wind_dir_x, params.wind_dir_y);
            let world_pos = vec2<f32>(f32(x) * dx, f32(y) * dy);
            let turb_wind = calc_wind_turbulence(world_pos, params.time, mean_dir, mean_wind_speed, params.wind_turbulence);

            // Conical wake plume orographic sheltering with dynamic turbulent meandering
            let shelter = calc_orographic_shelter_swe(x, y, z_c, dx, turb_wind, params.wind_shelter);

            let eff_wind = turb_wind * shelter;
            let eff_speed = length(eff_wind);
            if (eff_speed > 0.01) {
                let wind_accel = params.wind_drag_coeff * eff_speed * eff_wind / max(h_next, 0.08);
                raw_u += dt * wind_accel.x;
                raw_v += dt * wind_accel.y;
            }
        }

        let raw_speed = sqrt(raw_u * raw_u + raw_v * raw_v);
        let max_speed = 20.0;
        if (raw_speed > max_speed) {
            let scale = max_speed / raw_speed;
            raw_u *= scale;
            raw_v *= scale;
        }

        // Semi-implicit Manning friction drag
        let speed = sqrt(raw_u * raw_u + raw_v * raw_v);
        let manning_n = 0.025;
        let h_eff = max(h_next, h_dry);
        let drag = dt * G * manning_n * manning_n * speed / pow(h_eff, 4.0 / 3.0);
        let friction_factor = 1.0 / (1.0 + drag);
        u_next = raw_u * friction_factor;
        v_next = raw_v * friction_factor;
    }
 
    // Stream source injection (top center)
    if (params.stream_inflow_active > 0.5) {
        let inflow_x_center = i32(width / 2u);
        let inflow_radius = max(3, i32(f32(width) * 0.03));
        let inflow_y_end = max(4, i32(f32(height) * 0.04));
        if (y >= 1u && i32(y) <= inflow_y_end && abs(i32(x) - inflow_x_center) <= inflow_radius) {
            let target_h = max(0.6, 2.6 - z_c);
            if (h_next < target_h) {
                h_next = target_h;
            }
        }
    }

    // Coastal sink: absorb fluid entering bottom boundary into the ocean
    if (params.coastal_sink_active > 0.5) {
        let h_start = height - 12u;
        if (y >= h_start && y < height - 1u && x >= 1u && x < width - 1u) {
            h_next = h_next * 0.82;
        }
    }

    out_h[idx] = h_next;
    out_u[idx] = u_next;
    out_v[idx] = v_next;
    out_z[idx] = z_c;
}

@compute @workgroup_size(16, 16)
fn boundary(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = domain.grid_res_x;
    let height = domain.grid_res_y;

    if (x >= width || y >= height) { return; }

    let is_edge = x == 0u || x == width - 1u || y == 0u || y == height - 1u;
    if (!is_edge) { return; }

    let idx = get_idx(x, y);

    // 1. Four corners: diagonal reflection (handled first to avoid race conditions)
    if (x == 0u && y == 0u) {
        let idx_in = get_idx(1u, 1u);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = -out_u[idx_in];
        out_v[idx] = -out_v[idx_in];
    } else if (x == width - 1u && y == 0u) {
        let idx_in = get_idx(width - 2u, 1u);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = -out_u[idx_in];
        out_v[idx] = -out_v[idx_in];
    } else if (x == 0u && y == height - 1u) {
        let idx_in = get_idx(1u, height - 2u);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = -out_u[idx_in];
        out_v[idx] = -out_v[idx_in];
    } else if (x == width - 1u && y == height - 1u) {
        let idx_in = get_idx(width - 2u, height - 2u);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = -out_u[idx_in];
        out_v[idx] = -out_v[idx_in];
    } else if (x == 0u) {
        // West Boundary
        let idx_in = get_idx(1u, y);
        out_z[idx] = out_z[idx_in];
        if (params.west_type == 1u) {
            let factor = 1.0 - params.west_outflow_rate;
            out_h[idx] = out_h[idx_in] * factor;
            let u_in = out_u[idx_in];
            out_u[idx] = select(0.0, u_in * factor, u_in < 0.0);
            out_v[idx] = out_v[idx_in] * factor;
        } else {
            out_h[idx] = out_h[idx_in];
            out_u[idx] = -out_u[idx_in];
            out_v[idx] = out_v[idx_in];
        }
    } else if (x == width - 1u) {
        // East Boundary
        let idx_in = get_idx(width - 2u, y);
        out_z[idx] = out_z[idx_in];
        if (params.east_type == 1u) {
            let factor = 1.0 - params.east_outflow_rate;
            out_h[idx] = out_h[idx_in] * factor;
            let u_in = out_u[idx_in];
            out_u[idx] = select(0.0, u_in * factor, u_in > 0.0);
            out_v[idx] = out_v[idx_in] * factor;
        } else {
            out_h[idx] = out_h[idx_in];
            out_u[idx] = -out_u[idx_in];
            out_v[idx] = out_v[idx_in];
        }
    } else if (y == 0u) {
        // North Boundary
        let idx_in = get_idx(x, 1u);
        out_z[idx] = out_z[idx_in];
        if (params.north_type == 2u) {
            out_h[idx] = params.north_inflow_h;
            out_u[idx] = 0.0;
            out_v[idx] = params.north_inflow_v;
        } else if (params.north_type == 1u) {
            let factor = 1.0 - params.north_outflow_rate;
            out_h[idx] = out_h[idx_in] * factor;
            out_u[idx] = out_u[idx_in] * factor;
            let v_in = out_v[idx_in];
            out_v[idx] = select(0.0, v_in * factor, v_in < 0.0);
        } else {
            out_h[idx] = out_h[idx_in];
            out_u[idx] = out_u[idx_in];
            out_v[idx] = -out_v[idx_in];
        }
    } else if (y == height - 1u) {
        // South Boundary
        let idx_in = get_idx(x, height - 2u);
        let z_bed = out_z[idx_in];
        out_z[idx] = z_bed;
        if (params.south_type == 3u) {
            // Wave Generator
            let pi = 3.14159265;
            let tide_period = max(1e-4, params.south_tide_period);
            let tide_z = params.south_tide_amp * sin(2.0 * pi * params.time / tide_period);

            let wave_period = max(1e-4, params.south_wave_period);
            let phase = fract(params.time / wave_period);

            // Asymmetric coastal surge profile:
            // 1. Long sustained surge: 40% of cycle (e.g. 6.0s at T=15s)
            // 2. Receding backwash: 30% of cycle (e.g. 4.5s at T=15s)
            // 3. Calm inter-surge interval: 30% of cycle (e.g. 4.5s at T=15s)
            var wave_surge = 0.0;
            if (phase < 0.40) {
                let s = phase / 0.40;
                wave_surge = pow(sin(s * pi), 1.2);
            } else {
                let r = (phase - 0.40) / 0.60;
                if (r < 0.50) {
                    wave_surge = -0.35 * pow(sin((r / 0.50) * pi), 1.2);
                } else {
                    wave_surge = 0.0;
                }
            }

            let target_eta = params.south_base_eta + tide_z + params.south_wave_amp * wave_surge;
            let target_h = max(0.0, target_eta - z_bed);

            if (wave_surge > 0.01) {
                // Surge forward (towards North, so negative v)
                out_h[idx] = target_h;
                out_u[idx] = 0.0;
                out_v[idx] = -params.south_surge_speed * wave_surge;
            } else {
                // Backwash receding into ocean
                let v_in = out_v[idx_in];
                out_h[idx] = min(out_h[idx_in], target_h);
                out_u[idx] = out_u[idx_in] * 0.8;
                out_v[idx] = select(0.0, v_in, v_in > 0.0);
            }
        } else if (params.south_type == 1u) {
            let factor = 1.0 - params.south_outflow_rate;
            out_h[idx] = out_h[idx_in] * factor;
            out_u[idx] = out_u[idx_in] * factor;
            let v_in = out_v[idx_in];
            out_v[idx] = select(0.0, v_in * factor, v_in > 0.0);
        } else {
            out_h[idx] = out_h[idx_in];
            out_u[idx] = out_u[idx_in];
            out_v[idx] = -out_v[idx_in];
        }
    }
}
