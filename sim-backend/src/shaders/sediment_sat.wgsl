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

// 3 storage buffers: 2 read + 1 write
@group(0) @binding(2) var<storage, read> in_h: array<f32>;
@group(0) @binding(3) var<storage, read> in_sat: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_sat: array<f32>;

fn get_idx(x: u32, y: u32) -> u32 {
    return y * domain.grid_res_x + x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = domain.grid_res_x;
    let height = domain.grid_res_y;

    if (x == 0u || x >= width - 1u || y == 0u || y >= height - 1u) {
        return;
    }

    let idx = get_idx(x, y);
    let h = in_h[idx];

    if (h > 1e-4) {
        out_sat[idx] = 1.0;
    } else {
        let sat_c = in_sat[idx];
        let sat_l = in_sat[get_idx(x - 1u, y)];
        let sat_r = in_sat[get_idx(x + 1u, y)];
        let sat_t = in_sat[get_idx(x, y - 1u)];
        let sat_b = in_sat[get_idx(x, y + 1u)];

        let laplacian = sat_l + sat_r + sat_t + sat_b - 4.0 * sat_c;
        let diffusion_rate = 0.05;
        let drying_rate = 0.02;
        let next_sat = sat_c + (diffusion_rate * laplacian - drying_rate * sat_c) * dt;
        out_sat[idx] = clamp(next_sat, 0.0, 1.0);
    }
}
