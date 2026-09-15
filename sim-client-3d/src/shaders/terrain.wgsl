struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    domain_extent: vec2<f32>,
    time: f32,
    pad: f32,
    wind: vec4<f32>, // [dir_x, dir_y, speed, chop_factor]
    wind_turb: vec4<f32>, // [turbulence, shelter, caustics_boost, pad]
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

fn calc_wind_turbulence(world_pos: vec2<f32>, time: f32, mean_dir: vec2<f32>, mean_speed: f32, turb_scale: f32) -> vec2<f32> {
    if (turb_scale <= 0.001 || mean_speed <= 0.01) {
        return mean_dir * mean_speed;
    }
    let adv_pos = world_pos - mean_dir * (mean_speed * time * 0.65);

    // Large-scale atmospheric gust intermittency envelope G(x, t) in [0.35, 1.50]
    let env_arg1 = 0.045 * adv_pos.x + 0.038 * adv_pos.y;
    let env_arg2 = 0.041 * adv_pos.y - 0.032 * adv_pos.x + 0.7;
    let gust_raw = 0.5 + 0.5 * sin(env_arg1) * cos(env_arg2);
    let gust_envelope = 0.35 + 1.15 * smoothstep(0.25, 0.75, gust_raw);

    // Multi-rotor golden-ratio coordinate rotations
    let p1 = vec2<f32>(0.796 * adv_pos.x + 0.605 * adv_pos.y, -0.605 * adv_pos.x + 0.796 * adv_pos.y);
    let p2 = vec2<f32>(-0.255 * adv_pos.x + 0.967 * adv_pos.y, -0.967 * adv_pos.x - 0.255 * adv_pos.y);
    let p3 = vec2<f32>(-0.974 * adv_pos.x + 0.228 * adv_pos.y, -0.228 * adv_pos.x - 0.974 * adv_pos.y);

    let u1 = 0.11 * p1.x + cos(0.09 * p1.y);
    let u2 = 0.27 * p2.x - 0.23 * p2.y + 1.4;
    let u3 = 0.68 * p3.x + 0.54 * p3.y - 0.8;

    let du1_dx = 0.11 * 0.796 + 0.09 * sin(0.09 * p1.y) * 0.605;
    let du1_dy = 0.11 * 0.605 - 0.09 * sin(0.09 * p1.y) * 0.796;
    let du2_dx = -0.2913;
    let du2_dy = 0.2024;
    let du3_dx = -0.7854;
    let du3_dy = -0.3709;

    let dpsi_dx = cos(u1) * du1_dx + 0.55 * cos(u2) * du2_dx - 0.28 * sin(u3) * du3_dx;
    let dpsi_dy = cos(u1) * du1_dy + 0.55 * cos(u2) * du2_dy - 0.28 * sin(u3) * du3_dy;

    let curl_vec = vec2<f32>(dpsi_dy, -dpsi_dx);
    let effective_turb = turb_scale * gust_envelope * mean_speed * 0.75;
    return mean_dir * mean_speed + curl_vec * effective_turb;
}

fn calc_orographic_shelter_terrain(uv: vec2<f32>, turb_wind: vec2<f32>, shelter_strength: f32) -> f32 {
    if (shelter_strength <= 0.001) {
        return 1.0;
    }
    let turb_len = length(turb_wind);
    if (turb_len <= 0.01) {
        return 1.0;
    }
    let upwind_dir = -turb_wind / turb_len;
    let extent = camera.domain_extent;

    // 3-Ray Angular Wake Fan (+-15 degrees lateral spreading for conical wake)
    let dir_c = upwind_dir;
    let dir_l = vec2<f32>(upwind_dir.x * 0.966 - upwind_dir.y * 0.259, upwind_dir.x * 0.259 + upwind_dir.y * 0.966);
    let dir_r = vec2<f32>(upwind_dir.x * 0.966 + upwind_dir.y * 0.259, -upwind_dir.x * 0.259 + upwind_dir.y * 0.966);

    let d1 = 1.8 / extent;
    let d2 = 4.0 / extent;
    let d3 = 7.5 / extent;

    let z_c = textureSample(tex_elevation, samp, uv).r;
    let z_c1 = textureSample(tex_elevation, samp, uv + dir_c * d1).r;
    let z_c2 = textureSample(tex_elevation, samp, uv + dir_c * d2).r;
    let z_c3 = textureSample(tex_elevation, samp, uv + dir_c * d3).r;
    let dz_c = max(0.0, max(z_c1, max(z_c2, z_c3)) - z_c);

    let z_l1 = textureSample(tex_elevation, samp, uv + dir_l * d1).r;
    let z_l2 = textureSample(tex_elevation, samp, uv + dir_l * d2).r;
    let z_l3 = textureSample(tex_elevation, samp, uv + dir_l * d3).r;
    let dz_l = max(0.0, max(z_l1, max(z_l2, z_l3)) - z_c);

    let z_r1 = textureSample(tex_elevation, samp, uv + dir_r * d1).r;
    let z_r2 = textureSample(tex_elevation, samp, uv + dir_r * d2).r;
    let z_r3 = textureSample(tex_elevation, samp, uv + dir_r * d3).r;
    let dz_r = max(0.0, max(z_r1, max(z_r2, z_r3)) - z_c);

    let effective_dz = 0.50 * dz_c + 0.25 * dz_l + 0.25 * dz_r;
    let edge_jitter = 0.04 * sin(42.0 * uv.x + 37.0 * uv.y);
    let shelter_val = exp(-2.0 * shelter_strength * max(0.0, effective_dz + edge_jitter));
    return clamp(shelter_val, 0.02, 1.0);
}

@group(1) @binding(0) var tex_elevation: texture_2d<f32>;
@group(1) @binding(1) var tex_water: texture_2d<f32>;
@group(1) @binding(2) var tex_sed_sat: texture_2d<f32>;
@group(1) @binding(3) var samp: sampler;

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

    // Sample water depth to evaluate underwater optics & dynamic caustics
    let water_sample = textureSample(tex_water, samp, uv);
    let water_depth = water_sample.r;

    if (water_depth > 0.005) {
        // A. Beer-Lambert underwater light absorption on sand albedo
        let t_r = exp(-2.6 * water_depth);
        let t_g = exp(-0.70 * water_depth);
        let t_b = exp(-0.18 * water_depth);
        let water_tint = vec3<f32>(t_r, t_g, t_b);
        lit_color = lit_color * water_tint;

        // B. Dynamic Optical Caustics (Refracted sunlight convergence with organic turbulence & sheltering)
        let wind_len = length(camera.wind.xy);
        let wind_dir = select(vec2<f32>(0.0, -1.0), camera.wind.xy / wind_len, wind_len > 1e-4);
        let wind_speed = camera.wind.z;
        let chop_scale = camera.wind.w;
        let turb_scale = camera.wind_turb.x;
        let shelter_strength = camera.wind_turb.y;
        let caustics_boost = camera.wind_turb.z;
        let time = camera.time;

        // Local wind turbulence curl distortion on caustic UVs
        let turb_wind = calc_wind_turbulence(in.world_pos.xy, time, wind_dir, wind_speed, turb_scale);
        let turb_offset = (turb_wind - wind_dir * wind_speed) * (turb_scale * 0.25);

        // Conical wake plume orographic sheltering with dynamic turbulent meandering
        let shelter = calc_orographic_shelter_terrain(uv, turb_wind, shelter_strength);

        // Dual-layer moving wave phase coords organically distorted by turbulence
        let uv_caust1 = (in.world_pos.xy + turb_offset) * 2.2 + wind_dir * (time * 1.8);
        let uv_caust2 = (in.world_pos.xy - vec2<f32>(turb_offset.y, -turb_offset.x)) * 3.4 + vec2<f32>(-wind_dir.y, wind_dir.x) * (time * 1.2);

        // Wave surface curvature approximation for refracted light convergence
        let c1 = sin(uv_caust1.x + cos(uv_caust1.y * 1.3));
        let c2 = cos(uv_caust2.y + sin(uv_caust2.x * 1.3));
        let caustic_pattern = pow(max(0.0, c1 + c2), 3.0);

        // Depth attenuation: sharp in shallows, fading out as depth grows
        let depth_fade = exp(-0.90 * water_depth) * smoothstep(0.005, 0.04, water_depth);
        let eff_speed = wind_speed * shelter;
        let wind_amp = clamp(eff_speed / 5.0, 0.20, 2.2) * chop_scale * caustics_boost;
        let caustic_intensity = caustic_pattern * depth_fade * wind_amp * 1.40;

        let caustic_color = vec3<f32>(0.92, 0.98, 1.0) * sun_color;
        lit_color += caustic_color * caustic_intensity;
    } else {
        // C. Wet sand specular gloss ("Mirror Beach" on subaerial sand)
        if (sat > 0.35 && is_stone < 0.5) {
            let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);
            let half_vec = normalize(sun_dir + view_dir);
            let n_dot_h = max(dot(normal, half_vec), 0.0);
            let wet_spec = pow(n_dot_h, 32.0) * (sat - 0.35) * 1.5;
            lit_color += vec3<f32>(0.9, 0.95, 1.0) * wet_spec;
        }
    }

    return vec4<f32>(lit_color, 1.0);
}
