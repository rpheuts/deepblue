use glam::{Mat4, Vec2, Vec3, Vec4};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CameraMode {
    Perspective,
    Isometric, // True orthographic 2.5D view
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub domain_extent: [f32; 2],
    pub time: f32,
    pub pad: f32,
}

pub struct Camera {
    pub mode: CameraMode,
    pub target: Vec3,
    pub azimuth: f32,     // Horizontal orbit angle (radians)
    pub elevation: f32,   // Pitch angle above ground (radians)
    pub distance: f32,    // Distance from target
    pub aspect: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(domain_x: f32, domain_y: f32) -> Self {
        let center = Vec3::new(domain_x * 0.5, domain_y * 0.5, 0.5);
        let max_dim = domain_x.max(domain_y);
        Self {
            mode: CameraMode::Perspective,
            target: center,
            azimuth: std::f32::consts::FRAC_PI_4, // 45 degrees
            elevation: std::f32::consts::FRAC_PI_4 * 0.9, // ~40 degrees
            distance: max_dim * 1.55,
            aspect: 16.0 / 9.0,
            fov_y: 45.0f32.to_radians(),
            z_near: 0.1,
            z_far: 2000.0,
        }
    }

    pub fn position(&self) -> Vec3 {
        let cos_elev = self.elevation.cos();
        let sin_elev = self.elevation.sin();
        let sin_azim = self.azimuth.sin();
        let cos_azim = self.azimuth.cos();

        // Standard Z-up coordinate system: X = East, Y = North, Z = Up
        let offset = Vec3::new(
            self.distance * cos_elev * sin_azim,
            -self.distance * cos_elev * cos_azim,
            self.distance * sin_elev,
        );

        self.target + offset
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Z)
    }

    pub fn proj_matrix(&self) -> Mat4 {
        match self.mode {
            CameraMode::Perspective => {
                Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far)
            }
            CameraMode::Isometric => {
                let half_h = self.distance * (self.fov_y * 0.5).tan();
                let half_w = half_h * self.aspect;
                Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, self.z_near, self.z_far)
            }
        }
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    pub fn orbit(&mut self, delta_azim: f32, delta_elev: f32) {
        self.azimuth = (self.azimuth + delta_azim) % (std::f32::consts::TAU);
        let min_elev = 5.0f32.to_radians();
        let max_elev = 85.0f32.to_radians();
        self.elevation = (self.elevation + delta_elev).clamp(min_elev, max_elev);
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.position()).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        let up = right.cross(forward).normalize();

        let pan_speed = self.distance * 0.0018;
        self.target += right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);
    }

    pub fn zoom(&mut self, delta: f32) {
        let factor = if delta > 0.0 { 0.88 } else { 1.14 };
        self.distance = (self.distance * factor).clamp(2.0, 500.0);
    }

    pub fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            CameraMode::Perspective => CameraMode::Isometric,
            CameraMode::Isometric => CameraMode::Perspective,
        };
    }

    pub fn reset(&mut self, domain_x: f32, domain_y: f32) {
        let center = Vec3::new(domain_x * 0.5, domain_y * 0.5, 0.5);
        let max_dim = domain_x.max(domain_y);
        self.target = center;
        self.azimuth = std::f32::consts::FRAC_PI_4;
        self.elevation = std::f32::consts::FRAC_PI_4 * 0.9;
        self.distance = max_dim * 1.55;
    }

    /// Converts normalized screen pixel coordinates [0..w, 0..h] into a world-space Ray: (Origin, Direction)
    pub fn screen_to_ray(&self, screen_pos: Vec2, screen_size: Vec2) -> (Vec3, Vec3) {
        let ndc_x = (2.0 * screen_pos.x / screen_size.x) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.y / screen_size.y);

        let inv_vp = self.view_proj_matrix().inverse();

        let near_ndc = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_ndc = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near_world = inv_vp * near_ndc;
        let far_world = inv_vp * far_ndc;

        let ray_origin = near_world.truncate() / near_world.w;
        let ray_far = far_world.truncate() / far_world.w;
        let ray_dir = (ray_far - ray_origin).normalize();

        (ray_origin, ray_dir)
    }

    pub fn create_uniforms(&self, time: f32, domain_x: f32, domain_y: f32) -> CameraUniforms {
        let vp = self.view_proj_matrix();
        let inv_vp = vp.inverse();
        let pos = self.position();

        // Sunlight direction pointing downward-east (normalized)
        let sun = Vec3::new(-0.35, -0.45, 0.82).normalize();

        CameraUniforms {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: inv_vp.to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, 1.0],
            sun_dir: [sun.x, sun.y, sun.z, 0.0],
            domain_extent: [domain_x, domain_y],
            time,
            pad: 0.0,
        }
    }
}
