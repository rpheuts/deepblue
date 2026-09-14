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
@group(1) @binding(1) var tex_water: texture_2d<f32>;
@group(1) @binding(2) var tex_velocity: texture_2d<f32>;
@group(1) @binding(3) var tex_sed_sat: texture_2d<f32>;
@group(1) @binding(4) var samp: sampler;

struct VertexInput {
    @location(0) pos: vec2<f32>, // Normalized [0, 1] across domain
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) depth: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;

    let water = textureSampleLevel(tex_water, samp, in.uv, 0.0);
    let h = water.r;
    let eta = water.g; // z_bed + h
    out.depth = h;

    let world_x = in.pos.x * camera.domain_extent.x;
    let world_y = in.pos.y * camera.domain_extent.y;

    let world_pos = vec3<f32>(world_x, world_y, eta);

    out.world_pos = world_pos;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let water_sample = textureSample(tex_water, samp, uv);
    let depth = water_sample.r;
    if (depth <= 0.004) {
        discard;
    }

    let tex_dim = vec2<f32>(textureDimensions(tex_water));
    let texel_size = 1.0 / tex_dim;

    // Surface gradient of water elevation eta for 3D normal
    let eta_l = textureSample(tex_water, samp, uv - vec2<f32>(texel_size.x, 0.0)).g;
    let eta_r = textureSample(tex_water, samp, uv + vec2<f32>(texel_size.x, 0.0)).g;
    let eta_b = textureSample(tex_water, samp, uv - vec2<f32>(0.0, texel_size.y)).g;
    let eta_t = textureSample(tex_water, samp, uv + vec2<f32>(0.0, texel_size.y)).g;

    let cell_size_x = camera.domain_extent.x / tex_dim.x;
    let cell_size_y = camera.domain_extent.y / tex_dim.y;

    let deta_x = (eta_r - eta_l) / (2.0 * cell_size_x);
    let deta_y = (eta_t - eta_b) / (2.0 * cell_size_y);

    // Flow velocity for animated ripples and foam
    let vel_data = textureSample(tex_velocity, samp, uv);
    let u = vel_data.r;
    let v = vel_data.g;
    let speed = sqrt(u * u + v * v);

    // Micro-ripple wave normal distortion
    let ripple_freq = 18.0;
    let ripple_phase = (in.world_pos.x * 0.4 + in.world_pos.y * 0.4 - camera.time * 2.5) * ripple_freq;
    let ripple_dx = sin(ripple_phase) * 0.06;
    let ripple_dy = cos(ripple_phase) * 0.06;

    let normal = normalize(vec3<f32>(-(deta_x * 2.5 + ripple_dx), -(deta_y * 2.5 + ripple_dy), 1.0));

    // A. Physical Beer-Lambert Optical Extinction
    let t_r = exp(-4.2 * depth);
    let t_g = exp(-1.2 * depth);
    let t_b = exp(-0.35 * depth);

    let shallow_color = vec3<f32>(0.15, 0.72, 0.82);
    let deep_color = vec3<f32>(0.03, 0.18, 0.52);

    let extinction = vec3<f32>(t_r, t_g, t_b);
    var water_color = mix(deep_color, shallow_color, extinction.g);

    // B. Suspended Sediment Turbidity
    let sed_data = textureSample(tex_sed_sat, samp, uv);
    let c = clamp(sed_data.r, 0.0, 0.5);
    let turbidity = clamp(c / 0.08, 0.0, 1.0);
    let mud_color = vec3<f32>(0.65, 0.45, 0.25);
    water_color = mix(water_color, mud_color, turbidity);

    // C. Directional Lighting, Fresnel & Specular Sun Glint
    let sun_dir = normalize(camera.sun_dir.xyz);
    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);

    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = 0.03 + 0.97 * pow(1.0 - n_dot_v, 4.0);

    let sky_color = vec3<f32>(0.65, 0.82, 0.98);
    water_color = mix(water_color, sky_color, fresnel * 0.5);

    // Specular Sun Glint (Blinn-Phong)
    let half_vec = normalize(sun_dir + view_dir);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let sun_glint = pow(n_dot_h, 48.0) * 1.8;
    water_color += vec3<f32>(1.0, 0.98, 0.88) * sun_glint;

    // D. White Water Sea Foam (Rapids & Swash Breakers)
    var foam = 0.0;
    if (speed > 2.0) {
        foam = clamp((speed - 2.0) / 2.0, 0.0, 0.85);
    }
    if (depth < 0.06 && v < -0.08) {
        foam = max(foam, clamp((-v - 0.08) / 0.35, 0.0, 0.75));
    }

    let foam_color = vec3<f32>(0.96, 0.98, 1.0);
    let final_color = mix(water_color, foam_color, foam);
    let alpha = clamp(0.55 + depth * 0.85 + foam * 0.45, 0.45, 0.96) * smoothstep(0.004, 0.015, depth);

    return vec4<f32>(final_color, alpha);
}
