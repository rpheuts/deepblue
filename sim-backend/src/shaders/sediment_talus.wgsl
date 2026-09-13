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

// 4 storage buffers: 3 read + 1 write
@group(0) @binding(1) var<storage, read> in_z: array<f32>;
@group(0) @binding(2) var<storage, read> in_bedrock: array<f32>;
@group(0) @binding(3) var<storage, read> in_sat: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_z: array<f32>;

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
    let z_c = in_z[idx];
    let bedrock_c = in_bedrock[idx];
    let erodible_avail = max(0.0, z_c - bedrock_c);

    let dx = domain.extent_x / f32(width);
    let dy = domain.extent_y / f32(height);
    let sqrt2 = 1.41421356;

    let sat = clamp(in_sat[idx], 0.0, 1.0);
    // Dynamic critical slope:
    // dry = 34 deg (~0.6745), damp = 45 deg (1.000), saturated = 22 deg (~0.4040)
    var tan_crit = 0.6745;
    if (sat < 0.30) {
        tan_crit = 0.6745 + (1.000 - 0.6745) * (sat / 0.30);
    } else {
        tan_crit = 1.000 - (1.000 - 0.4040) * ((sat - 0.30) / 0.70);
    }

    var offsets_x = array<i32, 8>(-1, 1, 0, 0, -1, 1, -1, 1);
    var offsets_y = array<i32, 8>(0, 0, -1, 1, -1, -1, 1, 1);
    var dists = array<f32, 8>(dx, dx, dy, dy, dx * sqrt2, dx * sqrt2, dx * sqrt2, dx * sqrt2);

    // 1. Sand given away to lower neighbors
    var total_excess_out = 0.0;
    if (erodible_avail > 0.0) {
        for (var i = 0u; i < 8u; i = i + 1u) {
            let nx = i32(x) + offsets_x[i];
            let ny = i32(y) + offsets_y[i];
            if (nx >= 1 && nx < i32(width) - 1 && ny >= 1 && ny < i32(height) - 1) {
                let n_idx = get_idx(u32(nx), u32(ny));
                let dz = z_c - in_z[n_idx];
                let max_dz = dists[i] * tan_crit;
                if (dz > max_dz) {
                    total_excess_out = total_excess_out + (dz - max_dz);
                }
            }
        }
    }

    var loss = 0.0;
    if (total_excess_out > 0.0) {
        loss = min(erodible_avail * 0.45, total_excess_out * 0.25);
    }

    // 2. Sand received from higher neighbors
    var gain = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let nx = i32(x) + offsets_x[i];
        let ny = i32(y) + offsets_y[i];
        if (nx >= 1 && nx < i32(width) - 1 && ny >= 1 && ny < i32(height) - 1) {
            let n_idx = get_idx(u32(nx), u32(ny));
            let z_n = in_z[n_idx];
            let dz = z_n - z_c;
            let n_sat = clamp(in_sat[n_idx], 0.0, 1.0);
            var n_tan_crit = 0.6745;
            if (n_sat < 0.30) {
                n_tan_crit = 0.6745 + (1.000 - 0.6745) * (n_sat / 0.30);
            } else {
                n_tan_crit = 1.000 - (1.000 - 0.4040) * ((n_sat - 0.30) / 0.70);
            }

            let max_dz = dists[i] * n_tan_crit;
            if (dz > max_dz) {
                let n_avail = max(0.0, z_n - in_bedrock[n_idx]);
                let excess = dz - max_dz;
                let n_transfer = min(n_avail * 0.45, excess * 0.25) * (excess / (excess + 1e-4));
                gain = gain + (n_transfer * 0.125);
            }
        }
    }

    out_z[idx] = max(bedrock_c, z_c - loss + gain);
}
