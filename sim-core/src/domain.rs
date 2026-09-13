#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimDomainDescriptor {
    pub extent_x: f32,       // Domain width in real-world meters
    pub extent_y: f32,       // Domain length in real-world meters
    pub max_elevation: f32,  // Maximum allowable z_bed (meters)
    pub grid_res_x: u32,     // Discrete simulation grid width
    pub grid_res_y: u32,     // Discrete simulation grid height
    pub world_origin: [f32; 3], // 3D world space anchor [X, Y, Z]
}

impl Default for SimDomainDescriptor {
    fn default() -> Self {
        Self {
            extent_x: 100.0,
            extent_y: 100.0,
            max_elevation: 100.0,
            grid_res_x: 512,
            grid_res_y: 512,
            world_origin: [0.0, 0.0, 0.0],
        }
    }
}
