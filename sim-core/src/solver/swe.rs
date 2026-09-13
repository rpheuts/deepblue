use crate::state::DoubleBufferedGrid;

const G: f32 = 9.81;

/// Performs a single simulation step of the Shallow Water Equations.
/// Uses a first-order Rusanov (Local Lax-Friedrichs) finite volume scheme.
pub fn step_swe(grid: &mut DoubleBufferedGrid, dt: f32) {
    let dx = grid.descriptor.extent_x / grid.descriptor.grid_res_x as f32;
    let dy = grid.descriptor.extent_y / grid.descriptor.grid_res_y as f32;
    let width = grid.descriptor.grid_res_x;
    let height = grid.descriptor.grid_res_y;

    // Helper for wave speed
    let wave_speed = |h: f32, v: f32| -> f32 {
        v.abs() + (G * h).sqrt()
    };

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = grid.current.idx(x, y);
            
            // Current cell primitives
            let h_c = grid.current.h[idx];
            let u_c = grid.current.u[idx];
            let v_c = grid.current.v[idx];
            let z_c = grid.current.z_bed[idx];

            // Avoid computing dry cells to prevent div by zero
            if h_c < 1e-4 {
                // If it's totally dry and neighbors are dry, skip (optimization could be added here)
            }

            // Neighbor primitives
            let idx_l = grid.current.idx(x - 1, y);
            let idx_r = grid.current.idx(x + 1, y);
            let idx_t = grid.current.idx(x, y - 1);
            let idx_b = grid.current.idx(x, y + 1);

            let h_l = grid.current.h[idx_l];
            let h_r = grid.current.h[idx_r];
            let h_t = grid.current.h[idx_t];
            let h_b = grid.current.h[idx_b];
            
            let u_l = grid.current.u[idx_l];
            let u_r = grid.current.u[idx_r];
            let v_t = grid.current.v[idx_t];
            let v_b = grid.current.v[idx_b];

            let z_r = grid.current.z_bed[idx_r];
            let z_l = grid.current.z_bed[idx_l];
            let z_b = grid.current.z_bed[idx_b];
            let z_t = grid.current.z_bed[idx_t];

            // --- Hydrostatic Reconstruction (Audusse et al.) ---
            // Reconstruct h at interfaces to ensure Well-Balanced property (Lake at Rest)
            
            // X-interfaces
            let z_max_l = z_l.max(z_c);
            let h_l_xl = (h_l + z_l - z_max_l).max(0.0);
            let h_r_xl = (h_c + z_c - z_max_l).max(0.0);

            let z_max_r = z_c.max(z_r);
            let h_l_xr = (h_c + z_c - z_max_r).max(0.0);
            let h_r_xr = (h_r + z_r - z_max_r).max(0.0);

            // Y-interfaces
            let z_max_t = z_t.max(z_c);
            let h_l_yt = (h_t + z_t - z_max_t).max(0.0);
            let h_r_yt = (h_c + z_c - z_max_t).max(0.0);

            let z_max_b = z_c.max(z_b);
            let h_l_yb = (h_c + z_c - z_max_b).max(0.0);
            let h_r_yb = (h_b + z_b - z_max_b).max(0.0);

            // --- X Fluxes (F) ---
            let calc_f = |h: f32, u: f32, v: f32| -> (f32, f32, f32) {
                (h * u, h * u * u + 0.5 * G * h * h, h * u * v)
            };

            let f_l_xl = calc_f(h_l_xl, u_l, grid.current.v[idx_l]);
            let f_r_xl = calc_f(h_r_xl, u_c, v_c);
            
            let f_l_xr = calc_f(h_l_xr, u_c, v_c);
            let f_r_xr = calc_f(h_r_xr, u_r, grid.current.v[idx_r]);

            let a_x_l = wave_speed(h_l_xl, u_l).max(wave_speed(h_r_xl, u_c));
            let a_x_r = wave_speed(h_l_xr, u_c).max(wave_speed(h_r_xr, u_r));

            let flux_x_left_h = 0.5 * (f_l_xl.0 + f_r_xl.0) - 0.5 * a_x_l * (h_r_xl - h_l_xl);
            let flux_x_left_hu = 0.5 * (f_l_xl.1 + f_r_xl.1) - 0.5 * a_x_l * (h_r_xl * u_c - h_l_xl * u_l);
            let flux_x_left_hv = 0.5 * (f_l_xl.2 + f_r_xl.2) - 0.5 * a_x_l * (h_r_xl * v_c - h_l_xl * grid.current.v[idx_l]);

            let flux_x_right_h = 0.5 * (f_l_xr.0 + f_r_xr.0) - 0.5 * a_x_r * (h_r_xr - h_l_xr);
            let flux_x_right_hu = 0.5 * (f_l_xr.1 + f_r_xr.1) - 0.5 * a_x_r * (h_r_xr * u_r - h_l_xr * u_c);
            let flux_x_right_hv = 0.5 * (f_l_xr.2 + f_r_xr.2) - 0.5 * a_x_r * (h_r_xr * grid.current.v[idx_r] - h_l_xr * v_c);

            // --- Y Fluxes (G) ---
            let calc_g = |h: f32, u: f32, v: f32| -> (f32, f32, f32) {
                (h * v, h * u * v, h * v * v + 0.5 * G * h * h)
            };

            let g_l_yt = calc_g(h_l_yt, grid.current.u[idx_t], v_t);
            let g_r_yt = calc_g(h_r_yt, u_c, v_c);
            
            let g_l_yb = calc_g(h_l_yb, u_c, v_c);
            let g_r_yb = calc_g(h_r_yb, grid.current.u[idx_b], v_b);

            let a_y_t = wave_speed(h_l_yt, v_t).max(wave_speed(h_r_yt, v_c));
            let a_y_b = wave_speed(h_l_yb, v_c).max(wave_speed(h_r_yb, v_b));

            let flux_y_top_h = 0.5 * (g_l_yt.0 + g_r_yt.0) - 0.5 * a_y_t * (h_r_yt - h_l_yt);
            let flux_y_top_hu = 0.5 * (g_l_yt.1 + g_r_yt.1) - 0.5 * a_y_t * (h_r_yt * u_c - h_l_yt * grid.current.u[idx_t]);
            let flux_y_top_hv = 0.5 * (g_l_yt.2 + g_r_yt.2) - 0.5 * a_y_t * (h_r_yt * v_c - h_l_yt * v_t);

            let flux_y_bottom_h = 0.5 * (g_l_yb.0 + g_r_yb.0) - 0.5 * a_y_b * (h_r_yb - h_l_yb);
            let flux_y_bottom_hu = 0.5 * (g_l_yb.1 + g_r_yb.1) - 0.5 * a_y_b * (h_r_yb * grid.current.u[idx_b] - h_l_yb * u_c);
            let flux_y_bottom_hv = 0.5 * (g_l_yb.2 + g_r_yb.2) - 0.5 * a_y_b * (h_r_yb * v_b - h_l_yb * v_c);

            // --- Source Terms (Bed Slope) ---
            // Centered pressure force on the cell itself due to hydrostatic reconstruction
            let source_hu = 0.5 * G * (h_l_xr * h_l_xr - h_r_xl * h_r_xl) / dx;
            let source_hv = 0.5 * G * (h_l_yb * h_l_yb - h_r_yt * h_r_yt) / dy;

            // --- State Update ---
            let dh_dt = -((flux_x_right_h - flux_x_left_h) / dx + (flux_y_bottom_h - flux_y_top_h) / dy);
            let dhu_dt = -((flux_x_right_hu - flux_x_left_hu) / dx + (flux_y_bottom_hu - flux_y_top_hu) / dy) + source_hu;
            let dhv_dt = -((flux_x_right_hv - flux_x_left_hv) / dx + (flux_y_bottom_hv - flux_y_top_hv) / dy) + source_hv;

            let mut h_next = h_c + dh_dt * dt;
            let hu_next = (h_c * u_c) + dhu_dt * dt;
            let hv_next = (h_c * v_c) + dhv_dt * dt;

            // Positivity preserving and velocity recovery
            let mut u_next = 0.0;
            let mut v_next = 0.0;
            
            if h_next < 1e-4 {
                h_next = 0.0;
            } else {
                // Apply a small Manning-like friction to prevent velocities from diverging to infinity
                let friction = 0.999;
                u_next = (hu_next / h_next) * friction;
                v_next = (hv_next / h_next) * friction;
            }

            grid.next.h[idx] = h_next;
            grid.next.u[idx] = u_next;
            grid.next.v[idx] = v_next;
            grid.next.z_bed[idx] = z_c; // Bed static for hydrodynamics pass
        }
    }

    // Boundary conditions (Reflective / Wall)
    for x in 0..width {
        let idx_t0 = grid.next.idx(x, 0);
        let idx_t1 = grid.next.idx(x, 1);
        let idx_b0 = grid.next.idx(x, height - 1);
        let idx_b1 = grid.next.idx(x, height - 2);

        grid.next.h[idx_t0] = grid.next.h[idx_t1];
        grid.next.h[idx_b0] = grid.next.h[idx_b1];
        grid.next.z_bed[idx_t0] = grid.next.z_bed[idx_t1];
        grid.next.z_bed[idx_b0] = grid.next.z_bed[idx_b1];
        
        grid.next.u[idx_t0] = grid.next.u[idx_t1];
        grid.next.v[idx_t0] = -grid.next.v[idx_t1]; // Reflect
        
        grid.next.u[idx_b0] = grid.next.u[idx_b1];
        grid.next.v[idx_b0] = -grid.next.v[idx_b1]; // Reflect
    }
    
    for y in 0..height {
        let idx_l0 = grid.next.idx(0, y);
        let idx_l1 = grid.next.idx(1, y);
        let idx_r0 = grid.next.idx(width - 1, y);
        let idx_r1 = grid.next.idx(width - 2, y);

        grid.next.h[idx_l0] = grid.next.h[idx_l1];
        grid.next.h[idx_r0] = grid.next.h[idx_r1];
        grid.next.z_bed[idx_l0] = grid.next.z_bed[idx_l1];
        grid.next.z_bed[idx_r0] = grid.next.z_bed[idx_r1];

        grid.next.u[idx_l0] = -grid.next.u[idx_l1]; // Reflect
        grid.next.v[idx_l0] = grid.next.v[idx_l1];
        
        grid.next.u[idx_r0] = -grid.next.u[idx_r1]; // Reflect
        grid.next.v[idx_r0] = grid.next.v[idx_r1];
    }

    // Swap buffers for next tick
    grid.swap();
}
