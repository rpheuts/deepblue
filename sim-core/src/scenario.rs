use crate::boundary::{DomainBoundaryConfig, EdgeBoundary};
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

                // Non-erodible bedrock floor below erodible alluvium
                let bedrock = (z - 0.50).max(0.0);
                grid.current.bedrock_z[idx] = bedrock;
                grid.next.bedrock_z[idx] = bedrock;

                // Pre-fill upstream reservoir behind the dam with water up to spillway level
                if ny < 0.19 && dist < (channel_width * 2.5) {
                    let water_level = 2.45;
                    if water_level > z {
                        grid.current.h[idx] = water_level - z;
                    }
                }

                if grid.current.h[idx] > 1e-4 {
                    grid.current.soil_sat[idx] = 1.0;
                    grid.next.soil_sat[idx] = 1.0;
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

                grid.current.bedrock_z[idx] = (grid.current.z_bed[idx] - 0.40).max(0.0);
                grid.next.bedrock_z[idx] = grid.current.bedrock_z[idx];

                if grid.current.h[idx] > 1e-4 {
                    grid.current.soil_sat[idx] = 1.0;
                    grid.next.soil_sat[idx] = 1.0;
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
                grid.current.bedrock_z[idx] = 0.0;
                grid.next.bedrock_z[idx] = 0.0;
                grid.current.h[idx] = (target_eta - z).max(0.0);
                grid.current.soil_sat[idx] = 1.0;
                grid.next.soil_sat[idx] = 1.0;
            }
        }

        grid.apply_reflective_boundaries();
        grid
    }

    /// Creates a coastal beach scenario with incoming ocean swell/waves,
    /// a pre-built sandcastle with ramparts, corner towers, and moat in the swash zone,
    /// and a non-erodible stone breakwater / jetty demonstrating wave reflection and flow diversion.
    pub fn beach_sandcastle_waves(desc: SimDomainDescriptor) -> DoubleBufferedGrid {
        let mut grid = DoubleBufferedGrid::new(desc);
        let width = desc.grid_res_x;
        let height = desc.grid_res_y;

        let wave_gen = EdgeBoundary::WaveGenerator {
            base_elevation: 0.15,
            wave_amplitude: 0.46,
            wave_period: 25.0,
            surge_speed: 0.80,
            tide_amplitude: 0.12,
            tide_period: 90.0,
        };
        grid.boundaries = DomainBoundaryConfig::coastal_waves(wave_gen);

        let pi = std::f32::consts::PI;

        // Sandcastle center and size
        let castle_cx = 0.50f32;
        let castle_cy = 0.54f32;
        let castle_hw = 0.085f32; // half-width in normalized units (~8.5 meters)

        for y in 0..height {
            let ny = y as f32 / height as f32; // 0.0 (North/dunes) to 1.0 (South/ocean)

            for x in 0..width {
                let nx = x as f32 / width as f32; // 0.0 (West) to 1.0 (East)
                let idx = grid.current.idx(x, y);

                // --- 1. Base Coastline Bathymetry & Elevation ---
                // North (ny < 0.38): Dunes & backshore (z = 0.95m to 2.2m)
                // Middle (ny in 0.38..0.68): Steeper intertidal swash zone (z = 0.0m to 0.95m, increased pitch)
                // South (ny > 0.68): Deep offshore seabed (z = -1.45m to 0.0m, >2.3x deeper ocean)
                let mut z = if ny < 0.38 {
                    let dune_progress = (0.38 - ny) / 0.38;
                    0.95 + 1.25 * dune_progress.powf(1.2) + 0.08 * (nx * 6.0 * pi).sin() * (ny * 8.0 * pi).cos()
                } else if ny < 0.68 {
                    let beach_progress = (0.68 - ny) / 0.30;
                    beach_progress * 0.95 + 0.03 * (nx * 12.0 * pi).sin()
                } else {
                    let deep_progress = (ny - 0.68) / 0.32;
                    -1.45 * deep_progress.powf(0.85) + 0.03 * (nx * 8.0 * pi).cos()
                };

                // Default bedrock: 0.45m below ground level in dunes/beach, or deep in seabed
                let mut bedrock = (z - 0.45).max(-2.5);

                // --- 2. Pre-Built Stone Breakwater / Jetty (West Flank) ---
                // Extends from ny = 0.35 down to ny = 0.74, width ~ 3.5 meters
                let jetty_x = 0.22f32;
                let jetty_half_w = 0.018f32;
                if (nx - jetty_x).abs() <= jetty_half_w && (0.35..=0.74).contains(&ny) {
                    // Indestructible stone breakwater / groyne
                    let stone_height = 1.35f32;
                    z = z.max(stone_height);
                    bedrock = z; // bedrock == z_bed makes it indestructible stone!
                }

                // --- 3. Pre-Built Erodible Sandcastle (Center Swash Zone) ---
                let dx_c = (nx - castle_cx).abs();
                let dy_c = (ny - castle_cy).abs();
                let d_box = dx_c.max(dy_c);

                if d_box < castle_hw + 0.035 {
                    let base_beach_z = z;
                    // Keep bedrock below original beach level so all sandcastle features are erodible sand
                    bedrock = (base_beach_z - 0.25).max(-0.1);

                    // A. Moat: excavated ring surrounding the ramparts
                    if d_box >= castle_hw - 0.008 && d_box <= castle_hw + 0.024 {
                        let moat_depth = 0.22f32;
                        z = (base_beach_z - moat_depth).max(bedrock + 0.02);
                    }
                    // B. Outer Curtain Ramparts: sand embankment
                    else if d_box >= castle_hw - 0.032 && d_box < castle_hw - 0.008 {
                        let wall_h = 0.55f32;
                        z = base_beach_z + wall_h;
                    }
                    // C. Inner Courtyard & Central Keep
                    else if d_box < castle_hw - 0.032 {
                        let dist_center = (dx_c * dx_c + dy_c * dy_c).sqrt();
                        if dist_center < 0.022 {
                            // Central Keep Tower
                            z = base_beach_z + 0.85f32;
                        } else {
                            // Courtyard platform
                            z = base_beach_z + 0.22f32;
                        }
                    }

                    // D. Four Corner Bastion Towers
                    let corner_dist = ((dx_c - (castle_hw - 0.02)).powi(2) + (dy_c - (castle_hw - 0.02)).powi(2)).sqrt();
                    if corner_dist < 0.020 {
                        let tower_h = 0.75f32;
                        z = z.max(base_beach_z + tower_h);
                    }
                }

                grid.current.z_bed[idx] = z;
                grid.next.z_bed[idx] = z;
                grid.current.bedrock_z[idx] = bedrock;
                grid.next.bedrock_z[idx] = bedrock;

                // --- 4. Initial Water Level ---
                // Pre-fill ocean (still water level at eta = 0.15m)
                let still_water_eta = 0.15f32;
                if still_water_eta > z {
                    let h = still_water_eta - z;
                    grid.current.h[idx] = h;
                    grid.next.h[idx] = h;
                    grid.current.soil_sat[idx] = 1.0;
                    grid.next.soil_sat[idx] = 1.0;
                } else {
                    // Damp sand in swash zone, dry on upper dunes
                    let sat = if ny > 0.48 {
                        0.55 * ((ny - 0.48) / 0.20).clamp(0.0, 1.0)
                    } else {
                        0.05
                    };
                    grid.current.soil_sat[idx] = sat;
                    grid.next.soil_sat[idx] = sat;
                }
            }
        }

        grid.apply_boundaries();
        grid
    }
}
