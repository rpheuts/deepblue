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
@group(1) @binding(1) var samp: sampler;

struct VertexInput {
    @location(0) pos: vec3<f32>,    // [u, v, is_top]
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

const BASE_ELEVATION: f32 = -2.5;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let u = in.pos.x;
    let v = in.pos.y;
    let is_top = in.pos.z;

    let z_bed = textureSampleLevel(tex_elevation, samp, vec2<f32>(u, v), 0.0).r;
    let final_z = select(BASE_ELEVATION, z_bed, is_top > 0.5);

    let world_x = u * camera.domain_extent.x;
    let world_y = v * camera.domain_extent.y;
    let world_pos = vec3<f32>(world_x, world_y, final_z);

    out.world_pos = world_pos;
    out.normal = in.normal;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = normalize(camera.sun_dir.xyz);
    let normal = normalize(in.normal);

    // Architectural dark slate cross-section
    var base_color = vec3<f32>(0.22, 0.24, 0.27);

    // Subtle geological stratigraphy bands along Z
    let layer = sin(in.world_pos.z * 24.0);
    base_color += vec3<f32>(0.02, 0.015, 0.01) * layer;

    let n_dot_l = max(dot(normal, sun_dir), 0.0);
    let ambient = vec3<f32>(0.30, 0.35, 0.42);
    let sun_color = vec3<f32>(0.9, 0.88, 0.82);

    let lit = base_color * (ambient + sun_color * n_dot_l * 0.7);
    return vec4<f32>(lit, 1.0);
}
