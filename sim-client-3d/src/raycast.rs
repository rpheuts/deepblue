use glam::Vec3;

pub struct HeightfieldRaycaster;

impl HeightfieldRaycaster {
    /// Intersects a 3D world-space Ray (origin, dir) with the 2.5D heightfield bathymetry.
    /// Returns the world-space intersection point (Vec3) if hit, else None.
    pub fn intersect(
        ray_origin: Vec3,
        ray_dir: Vec3,
        extent_x: f32,
        extent_y: f32,
        grid_res_x: u32,
        grid_res_y: u32,
        z_bed: &[f32],
    ) -> Option<Vec3> {
        let min_z = -5.0;
        let max_z = 10.0;

        // Bounding box ray-AABB intersection
        let inv_d = Vec3::new(
            if ray_dir.x.abs() > 1e-6 { 1.0 / ray_dir.x } else { 1e6 },
            if ray_dir.y.abs() > 1e-6 { 1.0 / ray_dir.y } else { 1e6 },
            if ray_dir.z.abs() > 1e-6 { 1.0 / ray_dir.z } else { 1e6 },
        );

        let t0_x = (0.0 - ray_origin.x) * inv_d.x;
        let t1_x = (extent_x - ray_origin.x) * inv_d.x;
        let (t_min_x, t_max_x) = if t0_x < t1_x { (t0_x, t1_x) } else { (t1_x, t0_x) };

        let t0_y = (0.0 - ray_origin.y) * inv_d.y;
        let t1_y = (extent_y - ray_origin.y) * inv_d.y;
        let (t_min_y, t_max_y) = if t0_y < t1_y { (t0_y, t1_y) } else { (t1_y, t0_y) };

        let t0_z = (min_z - ray_origin.z) * inv_d.z;
        let t1_z = (max_z - ray_origin.z) * inv_d.z;
        let (t_min_z, t_max_z) = if t0_z < t1_z { (t0_z, t1_z) } else { (t1_z, t0_z) };

        let t_enter = t_min_x.max(t_min_y).max(t_min_z).max(0.0);
        let t_exit = t_max_x.min(t_max_y).min(t_max_z);

        if t_enter >= t_exit {
            return None;
        }

        let sample_z = |x: f32, y: f32| -> f32 {
            let u = (x / extent_x).clamp(0.0, 1.0);
            let v = (y / extent_y).clamp(0.0, 1.0);
            let gx = ((u * (grid_res_x - 1) as f32).round() as usize).min((grid_res_x - 1) as usize);
            let gy = ((v * (grid_res_y - 1) as f32).round() as usize).min((grid_res_y - 1) as usize);
            z_bed[gy * grid_res_x as usize + gx]
        };

        // Linear raymarching with 48 steps
        let steps = 48;
        let dt = (t_exit - t_enter) / steps as f32;
        let mut t_prev = t_enter;

        for i in 1..=steps {
            let t = t_enter + dt * (i as f32);
            let p = ray_origin + ray_dir * t;
            let terrain_z = sample_z(p.x, p.y);

            if p.z <= terrain_z {
                // Crossed beneath the terrain! Refine using 6 steps of bisection
                let mut lo = t_prev;
                let mut hi = t;
                for _ in 0..6 {
                    let mid = (lo + hi) * 0.5;
                    let p_mid = ray_origin + ray_dir * mid;
                    let z_mid = sample_z(p_mid.x, p_mid.y);
                    if p_mid.z <= z_mid {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let best_t = (lo + hi) * 0.5;
                let hit_pos = ray_origin + ray_dir * best_t;
                return Some(Vec3::new(hit_pos.x, hit_pos.y, sample_z(hit_pos.x, hit_pos.y)));
            }

            t_prev = t;
        }

        None
    }
}
