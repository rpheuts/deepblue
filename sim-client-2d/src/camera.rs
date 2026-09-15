#[derive(Copy, Clone, Debug)]
pub struct CameraState {
    pub target_x: f32, // normalized center [0.0, 1.0]
    pub target_y: f32,
    pub zoom: f32,     // 1.0 = full view, up to 6.0x
}

impl CameraState {
    pub fn new() -> Self {
        Self {
            target_x: 0.5,
            target_y: 0.5,
            zoom: 1.0,
        }
    }

    pub fn reset(&mut self) {
        self.target_x = 0.5;
        self.target_y = 0.5;
        self.zoom = 1.0;
    }

    /// Returns active view bounds in grid cells: (view_x, view_y, view_w, view_h)
    pub fn view_bounds(&self, width: f32, height: f32) -> (f32, f32, f32, f32) {
        let view_w = width / self.zoom;
        let view_h = height / self.zoom;
        let view_x = (self.target_x * width - view_w * 0.5).clamp(0.0, width - view_w);
        let view_y = (self.target_y * height - view_h * 0.5).clamp(0.0, height - view_h);
        (view_x, view_y, view_w, view_h)
    }

    /// Converts screen pixel coordinates to simulation grid cell coordinates
    pub fn screen_to_grid(&self, sx: f32, sy: f32, screen_w: f32, screen_h: f32, grid_w: f32, grid_h: f32) -> (i32, i32) {
        let (view_x, view_y, view_w, view_h) = self.view_bounds(grid_w, grid_h);
        let gx = (view_x + (sx / screen_w) * view_w) as i32;
        let gy = (view_y + (sy / screen_h) * view_h) as i32;
        (gx, gy)
    }

    /// Converts grid coordinates to screen pixel coordinates
    pub fn grid_to_screen(&self, gx: f32, gy: f32, screen_w: f32, screen_h: f32, grid_w: f32, grid_h: f32) -> (f32, f32) {
        let (view_x, view_y, view_w, view_h) = self.view_bounds(grid_w, grid_h);
        let sx = (gx - view_x) / view_w * screen_w;
        let sy = (gy - view_y) / view_h * screen_h;
        (sx, sy)
    }
}
