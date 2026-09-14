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

// Active storage buffers (read-only for current time step)
@group(0) @binding(1) var<storage, read> in_h: array<f32>;
@group(0) @binding(2) var<storage, read> in_u: array<f32>;
@group(0) @binding(3) var<storage, read> in_v: array<f32>;
@group(0) @binding(4) var<storage, read> in_z: array<f32>;
@group(0) @binding(5) var<storage, read> in_c: array<f32>;
@group(0) @binding(6) var<storage, read> in_sat: array<f32>;
@group(0) @binding(7) var<storage, read> in_bedrock: array<f32>;

// Output 2D Storage Textures (AGENTS.md contract)
// ElevationMap: R32Float -> z_bed
@group(0) @binding(8) var tex_elevation: texture_storage_2d<r32float, write>;

// WaterMap: Rgba32Float -> (h, eta = z_bed + h, speed = |v|, 1.0)
@group(0) @binding(9) var tex_water: texture_storage_2d<rgba32float, write>;

// VelocityMap: Rgba32Float -> (u, v, speed, 1.0)
@group(0) @binding(10) var tex_velocity: texture_storage_2d<rgba32float, write>;

// SedimentWetnessMap: Rgba32Float -> (C, W_sat, z_bedrock, is_stone = select(0.0, 1.0, z_bed <= z_bedrock + 1e-4))
@group(0) @binding(11) var tex_sed_sat: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = domain.grid_res_x;
    let height = domain.grid_res_y;

    if (x >= width || y >= height) {
        return;
    }

    let idx = y * width + x;
    let coords = vec2<i32>(i32(x), i32(y));

    let z = in_z[idx];
    let h = in_h[idx];
    let u = in_u[idx];
    let v = in_v[idx];
    let c = in_c[idx];
    let sat = in_sat[idx];
    let bedrock = in_bedrock[idx];

    let speed = sqrt(u * u + v * v);
    let eta = z + h;
    let is_stone = select(0.0, 1.0, z <= bedrock + 0.002);

    // 1. ElevationMap: R32Float (z_bed)
    textureStore(tex_elevation, coords, vec4<f32>(z, 0.0, 0.0, 1.0));

    // 2. WaterMap: Rgba32Float (h, eta, speed, 1.0)
    textureStore(tex_water, coords, vec4<f32>(h, eta, speed, 1.0));

    // 3. VelocityMap: Rgba32Float (u, v, speed, 1.0)
    textureStore(tex_velocity, coords, vec4<f32>(u, v, speed, 1.0));

    // 4. SedimentWetnessMap: Rgba32Float (C, W_sat, bedrock_z, is_stone)
    textureStore(tex_sed_sat, coords, vec4<f32>(c, sat, bedrock, is_stone));
}
