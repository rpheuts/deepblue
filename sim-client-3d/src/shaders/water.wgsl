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

struct GerstnerResult {
    displacement: vec3<f32>,
    normal_offset: vec2<f32>,
}

fn calc_wind_turbulence(world_pos: vec2<f32>, time: f32, mean_dir: vec2<f32>, mean_speed: f32, turb_scale: f32) -> vec2<f32> {
    if (turb_scale <= 0.001 || mean_speed <= 0.01) {
        return mean_dir * mean_speed;
    }
    // Taylor's frozen turbulence hypothesis: advect eddies downwind at wind speed
    let adv_pos = world_pos - mean_dir * (mean_speed * time * 0.65);

    // Large-scale atmospheric gust intermittency envelope G(x, t) in [0.35, 1.50]
    // Creates moving gust pockets ("cat's paws") separated by calm lulls
    let env_arg1 = 0.045 * adv_pos.x + 0.038 * adv_pos.y;
    let env_arg2 = 0.041 * adv_pos.y - 0.032 * adv_pos.x + 0.7;
    let gust_raw = 0.5 + 0.5 * sin(env_arg1) * cos(env_arg2);
    let gust_envelope = 0.35 + 1.15 * smoothstep(0.25, 0.75, gust_raw);

    // Multi-rotor golden-ratio coordinate rotations: prevents repeating lattice patterns
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

fn calc_orographic_shelter_vertex(uv: vec2<f32>, turb_wind: vec2<f32>, shelter_strength: f32) -> f32 {
    if (shelter_strength <= 0.001) {
        return 1.0;
    }
    let turb_len = length(turb_wind);
    if (turb_len <= 0.01) {
        return 1.0;
    }
    let upwind_dir = -turb_wind / turb_len;
    let extent = camera.domain_extent;

    let dir_c = upwind_dir;
    let dir_l = vec2<f32>(upwind_dir.x * 0.966 - upwind_dir.y * 0.259, upwind_dir.x * 0.259 + upwind_dir.y * 0.966);
    let dir_r = vec2<f32>(upwind_dir.x * 0.966 + upwind_dir.y * 0.259, -upwind_dir.x * 0.259 + upwind_dir.y * 0.966);

    let d1 = 1.8 / extent;
    let d2 = 4.0 / extent;
    let d3 = 7.5 / extent;

    let z_c = textureSampleLevel(tex_elevation, samp, uv, 0.0).r;
    let z_c1 = textureSampleLevel(tex_elevation, samp, uv + dir_c * d1, 0.0).r;
    let z_c2 = textureSampleLevel(tex_elevation, samp, uv + dir_c * d2, 0.0).r;
    let z_c3 = textureSampleLevel(tex_elevation, samp, uv + dir_c * d3, 0.0).r;
    let dz_c = max(0.0, max(z_c1, max(z_c2, z_c3)) - z_c);

    let z_l1 = textureSampleLevel(tex_elevation, samp, uv + dir_l * d1, 0.0).r;
    let z_l2 = textureSampleLevel(tex_elevation, samp, uv + dir_l * d2, 0.0).r;
    let z_l3 = textureSampleLevel(tex_elevation, samp, uv + dir_l * d3, 0.0).r;
    let dz_l = max(0.0, max(z_l1, max(z_l2, z_l3)) - z_c);

    let z_r1 = textureSampleLevel(tex_elevation, samp, uv + dir_r * d1, 0.0).r;
    let z_r2 = textureSampleLevel(tex_elevation, samp, uv + dir_r * d2, 0.0).r;
    let z_r3 = textureSampleLevel(tex_elevation, samp, uv + dir_r * d3, 0.0).r;
    let dz_r = max(0.0, max(z_r1, max(z_r2, z_r3)) - z_c);

    let effective_dz = 0.50 * dz_c + 0.25 * dz_l + 0.25 * dz_r;
    let edge_jitter = 0.04 * sin(42.0 * uv.x + 37.0 * uv.y);
    let shelter_val = exp(-2.0 * shelter_strength * max(0.0, effective_dz + edge_jitter));
    return clamp(shelter_val, 0.02, 1.0);
}

fn calc_orographic_shelter_fragment(uv: vec2<f32>, turb_wind: vec2<f32>, shelter_strength: f32) -> f32 {
    if (shelter_strength <= 0.001) {
        return 1.0;
    }
    let turb_len = length(turb_wind);
    if (turb_len <= 0.01) {
        return 1.0;
    }
    let upwind_dir = -turb_wind / turb_len;
    let extent = camera.domain_extent;

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

fn calc_gerstner(pos: vec2<f32>, time: f32, wind: vec4<f32>, wind_turb: vec4<f32>, shelter: f32) -> GerstnerResult {
    let wind_len = length(wind.xy);
    let d0_mean = select(vec2<f32>(0.0, -1.0), wind.xy / wind_len, wind_len > 1e-4);
    let speed_mean = wind.z;
    let chop_scale = wind.w;
    let turb_scale = wind_turb.x;

    var res: GerstnerResult;
    res.displacement = vec3<f32>(0.0);
    res.normal_offset = vec2<f32>(0.0);

    let eff_speed = speed_mean * shelter;
    let speed_mult = clamp(eff_speed / 8.0, 0.0, 2.5) * chop_scale;
    if (speed_mult < 1e-4) {
        return res;
    }

    // Evaluate local turbulent wind vector
    let turb_wind = calc_wind_turbulence(pos, time, d0_mean, speed_mean, turb_scale);
    let local_len = length(turb_wind);
    let d0 = select(d0_mean, turb_wind / local_len, local_len > 1e-4);

    let gust_factor = clamp(local_len / max(speed_mean, 0.1), 0.5, 1.8);

    // Multi-scale 3-tier incommensurate phase distortion: completely eliminates repeating regular patterns
    let p_adv = pos - d0 * (speed_mean * time * 0.40);
    let phase_jitter = turb_scale * (
        1.40 * sin(0.13 * p_adv.x + 0.17 * p_adv.y + 0.3 * time) +
        0.85 * sin(0.31 * p_adv.y - 0.27 * p_adv.x - 0.6 * time) +
        0.45 * cos(0.78 * p_adv.x + 0.65 * p_adv.y + 0.9 * time)
    );

    let total_amp = speed_mult * gust_factor;

    // 4 Directional Gerstner wave components with incommensurate wavelengths and angles
    // Octave 1: Main swell/wind sea (wavelength ~ 3.53m)
    let d1 = d0;
    let k1 = 2.0 * 3.14159 / 3.53;
    let c1 = sqrt(9.81 / k1);
    let a1 = 0.085 * total_amp;
    let q1 = 0.60;
    let phase1 = k1 * (dot(d1, pos) - c1 * time) + phase_jitter;
    let sin1 = sin(phase1);
    let cos1 = cos(phase1);

    // Octave 2: Angled chop (+31.4 degrees, wavelength ~ 2.07m)
    let d2 = vec2<f32>(d0.x * 0.853 - d0.y * 0.521, d0.x * 0.521 + d0.y * 0.853);
    let k2 = 2.0 * 3.14159 / 2.07;
    let c2 = sqrt(9.81 / k2);
    let a2 = 0.045 * total_amp;
    let q2 = 0.65;
    let phase2 = k2 * (dot(d2, pos) - c2 * time) - phase_jitter * 0.78;
    let sin2 = sin(phase2);
    let cos2 = cos(phase2);

    // Octave 3: Cross chop (-39.8 degrees, wavelength ~ 1.19m)
    let d3 = vec2<f32>(d0.x * 0.768 + d0.y * 0.640, -d0.x * 0.640 + d0.y * 0.768);
    let k3 = 2.0 * 3.14159 / 1.19;
    let c3 = sqrt(9.81 / k3);
    let a3 = 0.025 * total_amp;
    let q3 = 0.70;
    let phase3 = k3 * (dot(d3, pos) - c3 * time) + phase_jitter * 1.25;
    let sin3 = sin(phase3);
    let cos3 = cos(phase3);

    // Octave 4: High frequency capillary ripples (wavelength ~ 0.53m)
    let d4 = vec2<f32>(d0.x * 0.942 - d0.y * 0.336, d0.x * 0.336 + d0.y * 0.942);
    let k4 = 2.0 * 3.14159 / 0.53;
    let c4 = sqrt(9.81 / k4);
    let a4 = 0.012 * total_amp;
    let q4 = 0.75;
    let phase4 = k4 * (dot(d4, pos) - c4 * time) - phase_jitter * 1.62;
    let sin4 = sin(phase4);
    let cos4 = cos(phase4);

    let dx = q1 * a1 * d1.x * cos1 + q2 * a2 * d2.x * cos2 + q3 * a3 * d3.x * cos3 + q4 * a4 * d4.x * cos4;
    let dy = q1 * a1 * d1.y * cos1 + q2 * a2 * d2.y * cos2 + q3 * a3 * d3.y * cos3 + q4 * a4 * d4.y * cos4;
    let dz = a1 * sin1 + a2 * sin2 + a3 * sin3 + a4 * sin4;

    res.displacement = vec3<f32>(dx, dy, dz);

    let nx = -(d1.x * k1 * a1 * cos1 + d2.x * k2 * a2 * cos2 + d3.x * k3 * a3 * cos3 + d4.x * k4 * a4 * cos4);
    let ny = -(d1.y * k1 * a1 * cos1 + d2.y * k2 * a2 * cos2 + d3.y * k3 * a3 * cos3 + d4.y * k4 * a4 * cos4);
    res.normal_offset = vec2<f32>(nx, ny);

    return res;
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

    var world_pos = vec3<f32>(world_x, world_y, eta);

    // Displace vertices with Gerstner waves, smoothly fading near shoreline and in sheltered leeward zones
    let wave_weight = smoothstep(0.006, 0.06, h);
    if (wave_weight > 0.001) {
        let wind_len = length(camera.wind.xy);
        let d0_mean = select(vec2<f32>(0.0, -1.0), camera.wind.xy / wind_len, wind_len > 1e-4);
        let turb_wind = calc_wind_turbulence(vec2<f32>(world_x, world_y), camera.time, d0_mean, camera.wind.z, camera.wind_turb.x);
        let shelter = calc_orographic_shelter_vertex(in.uv, turb_wind, camera.wind_turb.y);

        let gerstner = calc_gerstner(vec2<f32>(world_x, world_y), camera.time, camera.wind, camera.wind_turb, shelter);
        world_pos.x += gerstner.displacement.x * wave_weight;
        world_pos.y += gerstner.displacement.y * wave_weight;
        world_pos.z += gerstner.displacement.z * wave_weight;
    }

    out.world_pos = world_pos;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

// 2D Hash function for fast procedural noise
fn hash21_sky(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 2D Value noise with Hermite cubic interpolation
fn noise2d_sky(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21_sky(i + vec2<f32>(0.0, 0.0));
    let b = hash21_sky(i + vec2<f32>(1.0, 0.0));
    let c = hash21_sky(i + vec2<f32>(0.0, 1.0));
    let d = hash21_sky(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Multi-octave Fractional Brownian Motion with irrational rotations
fn cloud_fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.52;
    var pos = p;
    for (var i = 0; i < 4; i = i + 1) {
        v += a * noise2d_sky(pos);
        pos = vec2<f32>(pos.x * 1.62 - pos.y * 1.21, pos.x * 1.21 + pos.y * 1.62) + vec2<f32>(1.7, 3.1);
        a *= 0.48;
    }
    return v;
}

// Evaluates the procedural sky dome, sun disc, corona, and dynamic drifting clouds
fn evaluate_sky(
    d: vec3<f32>,
    time: f32,
    sun_dir: vec3<f32>,
    wind: vec4<f32>,
) -> vec3<f32> {
    let sky_up = max(d.z, 0.0);
    let horizon_blend = smoothstep(0.005, 0.12, d.z);

    // 1. Sky Dome Gradient (warm coastal horizon to rich cerulean zenith)
    let horizon_color = vec3<f32>(0.76, 0.84, 0.94);
    let zenith_color = vec3<f32>(0.18, 0.40, 0.76);
    let sky_dome = mix(horizon_color, zenith_color, pow(sky_up, 0.50));

    // 2. Sun Disc & Solar Corona
    let sun_dot = max(dot(d, sun_dir), 0.0);
    let sun_disc = pow(sun_dot, 512.0) * 4.0;
    let sun_corona = pow(sun_dot, 28.0) * 0.40 * vec3<f32>(1.0, 0.96, 0.88);
    let sun_glow = pow(sun_dot, 5.0) * 0.20 * vec3<f32>(1.0, 0.88, 0.72);
    let sky_with_sun = sky_dome + vec3<f32>(1.0, 0.98, 0.92) * sun_disc + sun_corona + sun_glow;

    // 3. Procedural Cumulus Cloud Deck
    let cloud_dist = 1.0 / (sky_up + 0.14);
    let wind_speed = max(wind.z, 2.5);
    let wind_drift = wind.xy * (time * 0.006 * wind_speed);
    let cloud_uv = d.xy * cloud_dist * 0.28 + wind_drift;

    let cloud_raw = cloud_fbm(cloud_uv);

    // Soft cumulus billows with realistic clear sky breaks
    let coverage = 0.42;
    let cloud_density = smoothstep(coverage, coverage + 0.30, cloud_raw) * horizon_blend;

    // Self-shadowing from sun direction
    let sun_cloud_offset = normalize(sun_dir.xy) * 0.035;
    let cloud_sun_sample = cloud_fbm(cloud_uv + sun_cloud_offset);
    let self_shadow = clamp((cloud_sun_sample - cloud_raw) * 2.5, -0.3, 0.5);

    // Shaded base vs silver sunlit crests
    let cloud_shaded = vec3<f32>(0.66, 0.72, 0.82);
    let cloud_lit = vec3<f32>(1.0, 0.99, 0.98);
    var cloud_color = mix(cloud_lit, cloud_shaded, self_shadow + (1.0 - cloud_raw) * 0.35);

    // Forward Mie scattering (silver lining near sun)
    let silver_lining = pow(sun_dot, 10.0) * 0.80 * smoothstep(0.1, 0.7, cloud_raw);
    cloud_color += vec3<f32>(1.0, 0.96, 0.85) * silver_lining;

    var final_sky = mix(sky_with_sun, cloud_color, cloud_density * 0.90);

    // Below-horizon fade
    if (d.z <= 0.0) {
        let below_horizon_fade = clamp(-d.z * 3.5, 0.0, 1.0);
        let sea_ambient = vec3<f32>(0.07, 0.12, 0.22);
        final_sky = mix(horizon_color, sea_ambient, below_horizon_fade);
    }

    return final_sky;
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

    // Macro-scale surface gradient of water elevation eta
    let eta_l = textureSample(tex_water, samp, uv - vec2<f32>(texel_size.x, 0.0)).g;
    let eta_r = textureSample(tex_water, samp, uv + vec2<f32>(texel_size.x, 0.0)).g;
    let eta_b = textureSample(tex_water, samp, uv - vec2<f32>(0.0, texel_size.y)).g;
    let eta_t = textureSample(tex_water, samp, uv + vec2<f32>(0.0, texel_size.y)).g;

    let cell_size_x = camera.domain_extent.x / tex_dim.x;
    let cell_size_y = camera.domain_extent.y / tex_dim.y;

    let deta_x = (eta_r - eta_l) / (2.0 * cell_size_x);
    let deta_y = (eta_t - eta_b) / (2.0 * cell_size_y);

    // Conical wake plume orographic sheltering with dynamic turbulent meandering
    let wind_len = length(camera.wind.xy);
    let d0_mean = select(vec2<f32>(0.0, -1.0), camera.wind.xy / wind_len, wind_len > 1e-4);
    let turb_wind = calc_wind_turbulence(in.world_pos.xy, camera.time, d0_mean, camera.wind.z, camera.wind_turb.x);
    let shelter = calc_orographic_shelter_fragment(uv, turb_wind, camera.wind_turb.y);

    // Meso/Micro Gerstner wave normal perturbation with short-crested turbulence
    let gerstner = calc_gerstner(in.world_pos.xy, camera.time, camera.wind, camera.wind_turb, shelter);
    let wave_weight = smoothstep(0.006, 0.06, depth);

    // Flow velocity for foam advection and flow-map normal perturbation
    let vel_data = textureSample(tex_velocity, samp, uv);
    let u = vel_data.r;
    let v = vel_data.g;
    let speed = sqrt(u * u + v * v);

    // Flow-map driven normal perturbation: simulation velocity distorts micro-normals
    // Valve two-phase cycling prevents texture drift accumulation
    let flow_dir = vec2<f32>(u, v);
    let flow_speed = length(flow_dir);
    let flow_norm = select(vec2<f32>(0.0, 0.0), flow_dir / flow_speed, flow_speed > 0.01);
    let phase0 = fract(camera.time * 0.15);
    let phase1 = fract(camera.time * 0.15 + 0.5);
    let blend_t = abs(2.0 * phase0 - 1.0);
    let flow_scale = clamp(flow_speed * 0.12, 0.0, 0.8);
    let uv_flow0 = in.world_pos.xy * 0.8 + flow_norm * phase0 * flow_scale;
    let uv_flow1 = in.world_pos.xy * 0.8 + flow_norm * phase1 * flow_scale;
    let fn0 = vec2<f32>(
        sin(uv_flow0.x * 12.0 + cos(uv_flow0.y * 9.0)),
        cos(uv_flow0.y * 11.0 + sin(uv_flow0.x * 8.0))
    ) * 0.15;
    let fn1 = vec2<f32>(
        sin(uv_flow1.x * 12.0 + cos(uv_flow1.y * 9.0)),
        cos(uv_flow1.y * 11.0 + sin(uv_flow1.x * 8.0))
    ) * 0.15;
    let flow_normal_offset = mix(fn0, fn1, blend_t) * flow_scale;

    let total_nx = -(deta_x * 2.5) + gerstner.normal_offset.x * wave_weight * 1.5 + flow_normal_offset.x;
    let total_ny = -(deta_y * 2.5) + gerstner.normal_offset.y * wave_weight * 1.5 + flow_normal_offset.y;
    let normal = normalize(vec3<f32>(total_nx, total_ny, 1.0));

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

    // Procedural sky dome & dynamic drifting clouds reflection
    let reflect_dir = reflect(-view_dir, normal);
    let reflected_sky = evaluate_sky(reflect_dir, camera.time, sun_dir, camera.wind);
    water_color = mix(water_color, reflected_sky, fresnel * 0.70);

    // Specular Sun Glint (Blinn-Phong) across choppy wave facets
    let half_vec = normalize(sun_dir + view_dir);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let sun_glint = pow(n_dot_h, 48.0) * 2.2;
    water_color += vec3<f32>(1.0, 0.98, 0.88) * sun_glint;

    // E. Subsurface Scattering — translucent wave crest glow
    let sss_sun_through = max(dot(-sun_dir, view_dir), 0.0);
    let sss_depth_mask = exp(-8.0 * depth);
    let sss_wave_crest = max(gerstner.displacement.z * wave_weight, 0.0) * 6.0;
    let sss_intensity = pow(sss_sun_through, 3.0) * sss_depth_mask * sss_wave_crest;
    let sss_color = vec3<f32>(0.10, 0.75, 0.65);
    water_color += sss_color * sss_intensity * 0.8;

    // F. White Water Sea Foam (Rapids, Swash Breakers & Wind Whitecaps)
    let rapids_foam = clamp((speed - 1.8) / 2.5, 0.0, 1.0);
    let swash_foam = select(0.0, clamp((-v - 0.06) / 0.40, 0.0, 0.9), depth < 0.08 && v < -0.06);
    let whitecap_foam = select(0.0, clamp((gerstner.displacement.z - 0.04) / 0.03, 0.0, 0.7), depth > 0.2);
    var foam = max(rapids_foam, max(swash_foam, whitecap_foam));

    // Cellular foam dissolution texture
    if (foam > 0.01) {
        let foam_uv = in.world_pos.xy * 8.0 + vec2<f32>(u, v) * camera.time * 0.3;
        let cell1 = sin(foam_uv.x * 7.3 + foam_uv.y * 5.1 + camera.time * 2.0);
        let cell2 = cos(foam_uv.x * 11.7 - foam_uv.y * 8.3 - camera.time * 1.5);
        let cell_pattern = smoothstep(0.2, 0.8, cell1 * cell1 + cell2 * cell2);
        foam = foam * mix(0.5, 1.0, cell_pattern);
    }

    // Foam catches specular highlights
    let foam_spec = pow(n_dot_h, 12.0) * foam * 0.4;
    let foam_color = vec3<f32>(0.96, 0.98, 1.0) + vec3<f32>(1.0, 0.98, 0.90) * foam_spec;
    var final_color = mix(water_color, foam_color, foam);

    // Atmospheric depth haze
    let cam_dist = length(camera.camera_pos.xyz - in.world_pos);
    let haze = 1.0 - exp(-cam_dist * cam_dist * 0.00003);
    let haze_color = vec3<f32>(0.72, 0.80, 0.92);
    final_color = mix(final_color, haze_color, haze * 0.6);

    // Depth-dependent alpha with ultra-soft shoreline blend
    let shore_fade = smoothstep(0.003, 0.025, depth);
    let depth_opacity = 0.40 + 0.55 * (1.0 - exp(-3.5 * depth));
    let foam_opacity = foam * 0.50;
    let alpha = clamp(depth_opacity + foam_opacity, 0.0, 0.97) * shore_fade;

    return vec4<f32>(final_color, alpha);
}
