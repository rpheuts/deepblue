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

@group(0) @binding(0) var<uniform> domain: SimDomain;
@group(0) @binding(1) var<uniform> dt: f32;

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = domain.grid_res_x;
    let height = domain.grid_res_y;

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
        // Left boundary: invert normal velocity u, preserve tangential velocity v
        let idx_in = get_idx(1u, y);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = -out_u[idx_in];
        out_v[idx] = out_v[idx_in];
    } else if (x == width - 1u) {
        // Right boundary: invert normal velocity u, preserve tangential velocity v
        let idx_in = get_idx(width - 2u, y);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = -out_u[idx_in];
        out_v[idx] = out_v[idx_in];
    } else if (y == 0u) {
        // Top boundary: preserve tangential velocity u, invert normal velocity v
        let idx_in = get_idx(x, 1u);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = out_u[idx_in];
        out_v[idx] = -out_v[idx_in];
    } else if (y == height - 1u) {
        // Bottom boundary: preserve tangential velocity u, invert normal velocity v
        let idx_in = get_idx(x, height - 2u);
        out_h[idx] = out_h[idx_in];
        out_z[idx] = out_z[idx_in];
        out_u[idx] = out_u[idx_in];
        out_v[idx] = -out_v[idx_in];
    }
}
