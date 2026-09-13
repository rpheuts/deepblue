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

// 3 storage buffers: out_z, out_c, out_sat
@group(0) @binding(1) var<storage, read_write> out_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_c: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_sat: array<f32>;

fn get_idx(x: u32, y: u32) -> u32 {
    return y * domain.grid_res_x + x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = domain.grid_res_x;
    let height = domain.grid_res_y;

    if (x >= width || y >= height) { return; }

    let is_edge = x == 0u || x == width - 1u || y == 0u || y == height - 1u;
    if (!is_edge) { return; }

    let idx = get_idx(x, y);

    var src_x = x;
    var src_y = y;

    if (x == 0u) { src_x = 1u; }
    else if (x == width - 1u) { src_x = width - 2u; }

    if (y == 0u) { src_y = 1u; }
    else if (y == height - 1u) { src_y = height - 2u; }

    let src_idx = get_idx(src_x, src_y);

    out_z[idx] = out_z[src_idx];
    out_c[idx] = out_c[src_idx];
    out_sat[idx] = out_sat[src_idx];
}
