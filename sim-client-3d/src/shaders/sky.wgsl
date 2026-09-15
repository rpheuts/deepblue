struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    domain_extent: vec2<f32>,
    time: f32,
    pad: f32,
    wind: vec4<f32>,
    wind_turb: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_dir: vec3<f32>,
}

// 2D Hash function for fast procedural noise
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 2D Value noise with Hermite cubic interpolation
fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i + vec2<f32>(0.0, 0.0));
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Multi-octave Fractional Brownian Motion with irrational rotations
fn cloud_fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.52;
    var pos = p;
    for (var i = 0; i < 4; i = i + 1) {
        v += a * noise2d(pos);
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

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle covering NDC [-1, 1]
    let x = f32((in_vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(in_vertex_index & 2u) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);

    let unprojected = camera.inv_view_proj * vec4<f32>(x, y, 1.0, 1.0);
    out.world_dir = unprojected.xyz / unprojected.w - camera.camera_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = normalize(in.world_dir);
    let sun_dir = normalize(camera.sun_dir.xyz);
    let sky_color = evaluate_sky(ray_dir, camera.time, sun_dir, camera.wind);
    return vec4<f32>(sky_color, 1.0);
}
