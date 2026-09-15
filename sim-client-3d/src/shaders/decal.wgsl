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

@group(1) @binding(0) var tex_elevation: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct DecalUniforms {
    cursor_pos: vec4<f32>, // (x_world, y_world, radius_world, active_flag: 1.0 or 0.0)
    brush_color: vec4<f32>, // (r, g, b, alpha)
}

@group(2) @binding(0) var<uniform> decal: DecalUniforms;

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

    let z_bed = textureSampleLevel(tex_elevation, samp, in.uv, 0.0).r;
    let world_x = in.pos.x * camera.domain_extent.x;
    let world_y = in.pos.y * camera.domain_extent.y;

    // Elevate 15mm above terrain to avoid z-fighting
    let world_pos = vec3<f32>(world_x, world_y, z_bed + 0.015);

    out.world_pos = world_pos;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (decal.cursor_pos.w < 0.5) {
        discard;
    }

    let cursor_xy = decal.cursor_pos.xy;
    let radius = decal.cursor_pos.z;
    let dist = distance(in.world_pos.xy, cursor_xy);

    let ring_thickness = max(0.15, radius * 0.06);
    let outer_edge = radius;
    let inner_edge = radius - ring_thickness;

    if (dist > outer_edge + 0.2) {
        discard;
    }

    // Crisp anti-aliased ring edge
    let ring_val = smoothstep(outer_edge + 0.05, outer_edge, dist) *
                   smoothstep(inner_edge - 0.05, inner_edge, dist);

    // Subtle translucent interior fill
    let inner_fill = select(0.0, 0.18, dist <= inner_edge);

    let alpha = (ring_val * 0.85 + inner_fill) * decal.brush_color.a;
    if (alpha < 0.01) {
        discard;
    }

    // Additive glow on ring edge
    let ring_glow = ring_val * 0.4;
    let color = decal.brush_color.rgb + vec3<f32>(ring_glow);

    return vec4<f32>(color, alpha);
}
