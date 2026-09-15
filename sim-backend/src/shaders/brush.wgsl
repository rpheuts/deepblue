struct BrushParams {
    center_x: f32,
    center_y: f32,
    radius: f32,
    strength: f32,
    tool_type: u32,
    grid_res_x: u32,
    grid_res_y: u32,
    extent_x: f32,
    extent_y: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> brush: BrushParams;
@group(0) @binding(1) var<storage, read_write> buf_h0: array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_h1: array<f32>;
@group(0) @binding(3) var<storage, read_write> buf_z0: array<f32>;
@group(0) @binding(4) var<storage, read_write> buf_z1: array<f32>;
@group(0) @binding(5) var<storage, read_write> buf_bedrock: array<f32>;
@group(0) @binding(6) var<storage, read_write> buf_sat0: array<f32>;
@group(0) @binding(7) var<storage, read_write> buf_sat1: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = brush.grid_res_x;
    let height = brush.grid_res_y;

    if (x >= width || y >= height) {
        return;
    }

    let idx = y * width + x;

    let px = (f32(x) / f32(width - 1u)) * brush.extent_x;
    let py = (f32(y) / f32(height - 1u)) * brush.extent_y;

    let dx = px - brush.center_x;
    let dy = py - brush.center_y;
    let dist = sqrt(dx * dx + dy * dy);

    if (dist > brush.radius) {
        return;
    }

    let falloff = 1.0 - (dist / brush.radius);

    // Tool 1: Water Addition
    if (brush.tool_type == 1u) {
        let dh = brush.strength * 1.20 * falloff;
        buf_h0[idx] += dh;
        buf_h1[idx] += dh;
    }
    // Tool 2: Sand Dam Placement
    else if (brush.tool_type == 2u) {
        let dz = brush.strength * 0.60 * falloff;
        buf_z0[idx] += dz;
        buf_z1[idx] += dz;
        buf_sat0[idx] = 0.25;
        buf_sat1[idx] = 0.25;
    }
    // Tool 3: Stone Breakwater Construction
    else if (brush.tool_type == 3u) {
        let dz = brush.strength * 0.75 * falloff;
        let new_z = buf_z0[idx] + dz;
        buf_z0[idx] = new_z;
        buf_z1[idx] = new_z;
        buf_bedrock[idx] = new_z;
    }
    // Tool 4: Digging / Excavation
    else if (brush.tool_type == 4u) {
        let dz = brush.strength * 0.70 * falloff;
        let cur_z = buf_z0[idx];
        let bedrock = buf_bedrock[idx];
        let new_z = max(cur_z - dz, bedrock);
        buf_z0[idx] = new_z;
        buf_z1[idx] = new_z;
    }
}
