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

// 8 storage buffers: 6 read + 2 write
@group(0) @binding(2) var<storage, read> in_h: array<f32>;
@group(0) @binding(3) var<storage, read> in_u: array<f32>;
@group(0) @binding(4) var<storage, read> in_v: array<f32>;
@group(0) @binding(5) var<storage, read> in_z: array<f32>;
@group(0) @binding(6) var<storage, read> in_bedrock: array<f32>;
@group(0) @binding(7) var<storage, read> in_c: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_z: array<f32>;
@group(0) @binding(9) var<storage, read_write> out_c: array<f32>;

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
    let h_c = in_h[idx];
    let u_c = in_u[idx];
    let v_c = in_v[idx];
    let z_c = in_z[idx];
    let c_c = in_c[idx];
    let bedrock = in_bedrock[idx];

    let p = 0.40;
    let inv_one_minus_p = 1.0 / (1.0 - p);
    let ws = 0.04;
    let vc = 0.22;
    let k_cap = 0.05;
    let h_dry = 1e-4;

    let speed = sqrt(u_c * u_c + v_c * v_c);
    var c_eq = 0.0;
    if (speed > vc && h_c > h_dry) {
        let excess = speed * speed - vc * vc;
        c_eq = clamp(k_cap * excess / (1.0 + 0.5 * h_c), 0.0, 0.35);
    }

    let deposition = ws * c_c;
    var pickup = ws * c_eq;

    // Bedrock floor constraint: cannot erode below non-erodible bedrock
    let erodible_sand = max(0.0, z_c - bedrock);
    if (pickup > deposition) {
        let max_pickup = deposition + (erodible_sand * (1.0 - p)) / max(1e-5, dt);
        pickup = min(pickup, max_pickup);
    }

    var dz = (deposition - pickup) * inv_one_minus_p * dt;
    if (z_c + dz < bedrock) {
        dz = bedrock - z_c;
    }
    var z_next = z_c + dz;

    // Strict sediment conservation between bed and fluid column
    let net_source_fluid = -dz * (1.0 - p);
    let current_load = h_c * c_c;
    let next_load = max(0.0, current_load + net_source_fluid);

    var c_next = 0.0;
    if (h_c > h_dry) {
        let c_tentative = next_load / h_c;
        if (c_tentative > 0.50) {
            let max_load = 0.50 * h_c;
            let excess = next_load - max_load;
            z_next = z_next + excess * inv_one_minus_p;
            c_next = 0.50;
        } else {
            c_next = c_tentative;
        }
    } else {
        let excess_settle = next_load * inv_one_minus_p;
        z_next = z_next + excess_settle;
        c_next = 0.0;
    }

    out_z[idx] = z_next;
    out_c[idx] = c_next;
}
