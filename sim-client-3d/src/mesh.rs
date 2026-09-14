use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridVertex {
    pub pos: [f32; 2], // Normalized [0.0, 1.0] across domain
    pub uv: [f32; 2],  // Texture UV coordinates
}

impl GridVertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GridVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub struct GridMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    #[allow(dead_code)]
    pub res_x: u32,
    #[allow(dead_code)]
    pub res_y: u32,
}

impl GridMesh {
    /// Generates a static regular 2D grid patch of vertices and triangle indices.
    pub fn new(device: &wgpu::Device, res_x: u32, res_y: u32) -> Self {
        let mut vertices = Vec::with_capacity((res_x * res_y) as usize);
        let mut indices = Vec::with_capacity(((res_x - 1) * (res_y - 1) * 6) as usize);

        let inv_x = 1.0 / (res_x - 1) as f32;
        let inv_y = 1.0 / (res_y - 1) as f32;

        for y in 0..res_y {
            let v = y as f32 * inv_y;
            for x in 0..res_x {
                let u = x as f32 * inv_x;
                vertices.push(GridVertex {
                    pos: [u, v],
                    uv: [u, v],
                });
            }
        }

        for y in 0..(res_y - 1) {
            for x in 0..(res_x - 1) {
                let top_left = y * res_x + x;
                let top_right = top_left + 1;
                let bottom_left = (y + 1) * res_x + x;
                let bottom_right = bottom_left + 1;

                // Triangle 1 (CCW: top_left -> top_right -> bottom_left)
                indices.push(top_left);
                indices.push(top_right);
                indices.push(bottom_left);

                // Triangle 2 (CCW: top_right -> bottom_right -> bottom_left)
                indices.push(top_right);
                indices.push(bottom_right);
                indices.push(bottom_left);
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Mesh Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            res_x,
            res_y,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkirtVertex {
    pub pos: [f32; 3],    // [u, v, is_top: 1.0 or 0.0]
    pub normal: [f32; 3], // Outward surface normal
}

impl SkirtVertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SkirtVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct SkirtMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl SkirtMesh {
    /// Generates perimeter vertical walls around the domain boundary plus a base plate.
    pub fn new(device: &wgpu::Device, perimeter_res: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let inv_p = 1.0 / (perimeter_res - 1) as f32;

        let mut add_wall_quads = |start_u: f32, start_v: f32, end_u: f32, end_v: f32, normal: [f32; 3]| {
            let base_idx = vertices.len() as u32;
            for i in 0..perimeter_res {
                let t = i as f32 * inv_p;
                let u = start_u + (end_u - start_u) * t;
                let v = start_v + (end_v - start_v) * t;

                // Top vertex (displaced by bathymetry in vertex shader: is_top = 1.0)
                vertices.push(SkirtVertex {
                    pos: [u, v, 1.0],
                    normal,
                });
                // Bottom vertex (flat base plate at base depth: is_top = 0.0)
                vertices.push(SkirtVertex {
                    pos: [u, v, 0.0],
                    normal,
                });
            }

            for i in 0..(perimeter_res - 1) {
                let tl = base_idx + i * 2;
                let bl = tl + 1;
                let tr = tl + 2;
                let br = tr + 1;

                indices.push(tl);
                indices.push(bl);
                indices.push(tr);

                indices.push(tr);
                indices.push(bl);
                indices.push(br);
            }
        };

        // South wall: (x: 0->1, y: 0), normal = (0, -1, 0)
        add_wall_quads(0.0, 0.0, 1.0, 0.0, [0.0, -1.0, 0.0]);
        // East wall: (x: 1, y: 0->1), normal = (1, 0, 0)
        add_wall_quads(1.0, 0.0, 1.0, 1.0, [1.0, 0.0, 0.0]);
        // North wall: (x: 1->0, y: 1), normal = (0, 1, 0)
        add_wall_quads(1.0, 1.0, 0.0, 1.0, [0.0, 1.0, 0.0]);
        // West wall: (x: 0, y: 1->0), normal = (-1, 0, 0)
        add_wall_quads(0.0, 1.0, 0.0, 0.0, [-1.0, 0.0, 0.0]);

        // Bottom base plate quad
        let base_plate_start = vertices.len() as u32;
        vertices.push(SkirtVertex { pos: [0.0, 0.0, 0.0], normal: [0.0, 0.0, -1.0] });
        vertices.push(SkirtVertex { pos: [1.0, 0.0, 0.0], normal: [0.0, 0.0, -1.0] });
        vertices.push(SkirtVertex { pos: [1.0, 1.0, 0.0], normal: [0.0, 0.0, -1.0] });
        vertices.push(SkirtVertex { pos: [0.0, 1.0, 0.0], normal: [0.0, 0.0, -1.0] });

        indices.push(base_plate_start);
        indices.push(base_plate_start + 2);
        indices.push(base_plate_start + 1);

        indices.push(base_plate_start);
        indices.push(base_plate_start + 3);
        indices.push(base_plate_start + 2);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skirt Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skirt Mesh Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }
}
