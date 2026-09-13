use crate::domain::SimDomainDescriptor;
use crate::state::DoubleBufferedGrid;

/// Pre-configured hydraulic simulation scenarios for testing, demonstration, and visual inspection.
pub struct Scenarios;

impl Scenarios {
    /// Creates a dynamic beach stream / mountain valley scenario.
    ///
    /// Topography features:
    /// - Mountain valley sloping from z = 3.5m down to sea level z = 0.0m
    /// - An upstream reservoir basin pre-filled with water
    /// - An earthen dam ridge with a centered spillway notch
    /// - A winding canyon / river channel with downstream islands that induce flow splitting
    /// - Pre-configured continuous spring inflow zone and coastal sink zone
    pub fn beach_stream(desc: SimDomainDescriptor) -> DoubleBufferedGrid {
        let mut grid = DoubleBufferedGrid::new(desc);
        let width = desc.grid_res_x;
        let height = desc.grid_res_y;

        let pi = std::f32::consts::PI;

        for y in 0..height {
            let ny = y as f32 / height as f32; // [0.0, 1.0]

            // River channel centerline follows a natural sinusoidal meander
            let cx = 0.5 + 0.14 * (ny * 3.0 * pi).sin() + 0.06 * (ny * 5.0 * pi).cos();
            let channel_width = 0.07 + 0.08 * ny; // Widens as it heads to the sea

            // Base elevation: mountain ridge at top sloping down to the coast
            let z_base = 3.6 * (1.0 - ny).powf(1.15);

            for x in 0..width {
                let nx = x as f32 / width as f32;
                let idx = grid.current.idx(x, y);

                // Distance from river centerline
                let dist = (nx - cx).abs();

                // Channel carved gorge (Gaussian depression)
                let gorge_depth = 1.9 * (-((dist / channel_width).powi(2))).exp();
                let mut z = (z_base - gorge_depth).max(0.08);

                // Lateral valley walls rising on the sides
                let valley_wall = 1.5 * ((nx - 0.5).abs() * 2.0).powi(2);
                z += valley_wall;

                // Upstream dam at ny in [0.16, 0.22]
                let dam_profile = (-(((ny - 0.19) / 0.025).powi(2))).exp();
                if dam_profile > 0.01 {
                    let dam_crest = 1.6 * dam_profile;
                    // Spillway notch in the center of the dam
                    let notch_width = 0.035;
                    let notch_factor = (dist / notch_width).min(1.0);
                    let dam_height = dam_crest * (0.35 + 0.65 * notch_factor * notch_factor);
                    z += dam_height;
                }

                // Island 1 at ny = 0.45
                let island1_dist = ((nx - (cx - 0.03)).powi(2) + (ny - 0.45).powi(2)).sqrt();
                z += 0.9 * (-((island1_dist / 0.035).powi(2))).exp();

                // Island 2 at ny = 0.72
                let island2_dist = ((nx - (cx + 0.04)).powi(2) + (ny - 0.72).powi(2)).sqrt();
                z += 0.8 * (-((island2_dist / 0.04).powi(2))).exp();

                // Subtle organic ground bumps
                let noise = 0.04 * (nx * 50.0).sin() * (ny * 50.0).cos();
                z = (z + noise).max(0.02);

                grid.current.z_bed[idx] = z;
                grid.next.z_bed[idx] = z;

                // Pre-fill upstream reservoir behind the dam with water up to spillway level
                if ny < 0.19 && dist < (channel_width * 2.5) {
                    let water_level = 2.45;
                    if water_level > z {
                        grid.current.h[idx] = water_level - z;
                    }
                }
            }
        }

        grid.apply_reflective_boundaries();
        grid
    }

    /// Creates a classic 2D dam-break scenario with a reservoir held behind a central wall with a breach.
    pub fn dam_break(desc: SimDomainDescriptor) -> DoubleBufferedGrid {
        let mut grid = DoubleBufferedGrid::new(desc);
        let width = desc.grid_res_x;
        let height = desc.grid_res_y;

        let dam_x = (width as f32 * 0.4) as u32;
        let breach_y_start = (height as f32 * 0.4) as u32;
        let breach_y_end = (height as f32 * 0.6) as u32;

        for y in 0..height {
            for x in 0..width {
                let idx = grid.current.idx(x, y);
                // Mild slope downstream
                let z = (1.0 - (x as f32 / width as f32)) * 0.5;
                grid.current.z_bed[idx] = z;
                grid.next.z_bed[idx] = z;

                // High water reservoir on upstream side
                if x < dam_x {
                    grid.current.h[idx] = 4.0;
                } else if x == dam_x && (y < breach_y_start || y > breach_y_end) {
                    // Solid barrier wall
                    grid.current.z_bed[idx] = 6.0;
                    grid.next.z_bed[idx] = 6.0;
                }
            }
        }

        grid.apply_reflective_boundaries();
        grid
    }

    /// Creates an uneven bathymetry with a flat water surface for equilibrium verification.
    pub fn lake_at_rest(desc: SimDomainDescriptor) -> DoubleBufferedGrid {
        let mut grid = DoubleBufferedGrid::new(desc);
        let target_eta = 5.0f32;

        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = grid.current.idx(x, y);
                let z = 1.0 + (x as f32 * 0.1).sin() * 0.5 + (y as f32 * 0.1).cos() * 0.5;
                grid.current.z_bed[idx] = z;
                grid.next.z_bed[idx] = z;
                grid.current.h[idx] = (target_eta - z).max(0.0);
            }
        }

        grid.apply_reflective_boundaries();
        grid
    }
}
