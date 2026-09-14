struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    domain_extent: vec2<f32>,
    time: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var tex_elevation: texture_2d<f32>;
@group(1) @binding(1) var tex_sed_sat: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

struct VertexInput {
    @location(0) pos: vec2<f32>, // Normalized [0, 1] across domain
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;

    // Sample terrain elevation z_bed in meters
    let z_bed = textureSampleLevel(tex_elevation, samp, in.uv, 0.0).r;

    // Map normalized [0, 1] pos to physical world meters
    let world_x = in.pos.x * camera.domain_extent.x;
    let world_y = in.pos.y * camera.domain_extent.y;
    let world_pos = vec3<f32>(world_x, world_y, z_bed);

    out.world_pos = world_pos;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let tex_dim = vec2<f32>(textureDimensions(tex_elevation));
    let texel_size = 1.0 / tex_dim;

    // Finite-difference surface normal reconstruction from heightfield
    let z_l = textureSample(tex_elevation, samp, uv - vec2<f32>(texel_size.x, 0.0)).r;
    let z_r = textureSample(tex_elevation, samp, uv + vec2<f32>(texel_size.x, 0.0)).r;
    let z_b = textureSample(tex_elevation, samp, uv - vec2<f32>(0.0, texel_size.y)).r;
    let z_t = textureSample(tex_elevation, samp, uv + vec2<f32>(0.0, texel_size.y)).r;

    let cell_size_x = camera.domain_extent.x / tex_dim.x;
    let cell_size_y = camera.domain_extent.y / tex_dim.y;

    let dz_dx = (z_r - z_l) / (2.0 * cell_size_x);
    let dz_dy = (z_t - z_b) / (2.0 * cell_size_y);
    let normal = normalize(vec3<f32>(-dz_dx, -dz_dy, 1.0));

    // Sample soil wetness, turbidity, and stone classification
    // tex_sed_sat stores: (C, W_sat, bedrock_z, is_stone)
    let sed_data = textureSample(tex_sed_sat, samp, uv);
    let sat = clamp(sed_data.g, 0.0, 1.0);
    let is_stone = sed_data.a;

    // Slope calculation
    let slope = sqrt(dz_dx * dz_dx + dz_dy * dz_dy);

    // Material Albedo Splatting
    var albedo: vec3<f32>;
    if (is_stone > 0.5) {
        // Indestructible stone breakwater / masonry granite
        albedo = vec3<f32>(0.46, 0.49, 0.53);
    } else if (slope > 0.65) {
        // Steep canyon bedrock
        albedo = vec3<f32>(0.58, 0.52, 0.46);
    } else {
        // Golden beach sand
        albedo = vec3<f32>(0.84, 0.74, 0.54);
    }

    // Moisture darkening: wet sand darkens naturally
    let moisture_factor = 1.0 - 0.32 * sat;
    albedo = albedo * moisture_factor;

    // Directional sunlight + ambient sky lighting
    let sun_dir = normalize(camera.sun_dir.xyz);
    let n_dot_l = max(dot(normal, sun_dir), 0.0);
    let ambient = vec3<f32>(0.24, 0.30, 0.38);
    let sun_color = vec3<f32>(1.0, 0.96, 0.88);

    var lit_color = albedo * (ambient + sun_color * n_dot_l);

    // Wet sand specular gloss ("Mirror Beach")
    if (sat > 0.35 && is_stone < 0.5) {
        let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);
        let half_vec = normalize(sun_dir + view_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let wet_spec = pow(n_dot_h, 32.0) * (sat - 0.35) * 1.5;
        lit_color += vec3<f32>(0.9, 0.95, 1.0) * wet_spec;
    }

    return vec4<f32>(lit_color, 1.0);
}
