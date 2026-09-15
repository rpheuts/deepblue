use macroquad::prelude::Image;
use rayon::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::GridState;

/// Directional 3D hillshading pass executed in parallel across grid rows.
pub fn compute_hillshade_parallel(z_bed: &[f32], shade_cache: &mut [f32], width: usize, height: usize) {
    shade_cache
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let y_prev_offset = if y > 0 { (y - 1) * width } else { y * width };
            let y_next_offset = if y < height - 1 { (y + 1) * width } else { y * width };
            let y_curr_offset = y * width;

            for x in 0..width {
                let x_prev = if x > 0 { x - 1 } else { x };
                let x_next = if x < width - 1 { x + 1 } else { x };

                let dz_x = (z_bed[y_curr_offset + x_next] - z_bed[y_curr_offset + x_prev]) * 0.5;
                let dz_y = (z_bed[y_next_offset + x] - z_bed[y_prev_offset + x]) * 0.5;
                row[x] = (1.0 - 0.28 * (dz_x + dz_y)).clamp(0.60, 1.40);
            }
        });
}

/// High-fidelity CPU optics renderer executing parallel row blitting via Rayon.
/// Features physical Beer-Lambert extinction, deep body in-scattering, sediment turbidity,
/// 3D surface normal reconstruction, Blinn-Phong specular sun glints, Fresnel reflections,
/// multi-source wave/shore foam, and wet sand gloss ("mirror beach").
pub fn render_terrain_and_water(
    img: &mut Image,
    state: &GridState,
    shade_cache: &[f32],
    desc: &SimDomainDescriptor,
    is_coastal_waves: bool,
) {
    let width = desc.grid_res_x as usize;
    let height = desc.grid_res_y as usize;
    let row_bytes = width * 4;

    img.bytes
        .par_chunks_exact_mut(row_bytes)
        .enumerate()
        .for_each(|(y, row_slice)| {
            let row_offset = y * width;
            let y_prev_offset = if y > 0 { (y - 1) * width } else { y * width };
            let y_next_offset = if y < height - 1 { (y + 1) * width } else { y * width };

            for x in 0..width {
                let idx = row_offset + x;
                let byte_idx = x * 4;

                let z = state.z_bed[idx];
                let depth = state.h[idx];
                let sat = state.soil_sat[idx].clamp(0.0, 1.0);
                let shade = shade_cache[idx];

                let x_prev = if x > 0 { x - 1 } else { x };
                let x_next = if x < width - 1 { x + 1 } else { x };

                // 1. Terrain base material & elevation color
                let is_stone_rock = (z - state.bedrock_z[idx]).abs() < 0.04 && state.bedrock_z[idx] > 0.25;
                let (tr, tg, tb) = if is_stone_rock {
                    // Indestructible stone breakwater / masonry granite
                    (120.0, 125.0, 135.0)
                } else if z < 0.9 {
                    // Moist gravel / wet sand
                    (155.0, 130.0, 95.0)
                } else if z < 2.2 {
                    // Golden beach sand dunes
                    (215.0, 185.0, 135.0)
                } else {
                    // Rocky canyon wall
                    (160.0, 145.0, 130.0)
                };

                // Soil moisture: wet sand darkens naturally
                let moisture_darkening = 1.0 - 0.32 * sat;
                let r_land = (tr * shade * moisture_darkening).clamp(0.0, 255.0);
                let g_land = (tg * shade * moisture_darkening).clamp(0.0, 255.0);
                let b_land = (tb * shade * moisture_darkening).clamp(0.0, 255.0);

                if depth > 0.005 {
                    // --- HIGH-FIDELITY WATER OPTICS & LIGHTING PASS ---
                    let u = state.u[idx];
                    let v = state.v[idx];
                    let speed_sq = u * u + v * v;
                    let speed = speed_sq.sqrt();

                    // 3D Water Surface Normal & Slopes
                    let eta_l = state.z_bed[row_offset + x_prev] + state.h[row_offset + x_prev];
                    let eta_r = state.z_bed[row_offset + x_next] + state.h[row_offset + x_next];
                    let eta_t = state.z_bed[y_prev_offset + x] + state.h[y_prev_offset + x];
                    let eta_b = state.z_bed[y_next_offset + x] + state.h[y_next_offset + x];

                    let deta_x = (eta_r - eta_l) * 0.5;
                    let deta_y = (eta_b - eta_t) * 0.5;

                    // A. Physical Beer-Lambert Optical Extinction
                    let t_r = (-4.2 * depth).exp();
                    let t_g = (-1.25 * depth).exp();
                    let t_b = (-0.40 * depth).exp();

                    // Deep water body in-scattering
                    let deep_r = 10.0;
                    let deep_g = 58.0;
                    let deep_b = 148.0;
                    let inscatter_r = deep_r * (1.0 - t_r);
                    let inscatter_g = deep_g * (1.0 - t_g);
                    let inscatter_b = deep_b * (1.0 - t_b);

                    let mut water_r = r_land * t_r + inscatter_r;
                    let mut water_g = g_land * t_g + inscatter_g;
                    let mut water_b = b_land * t_b + inscatter_b;

                    // B. Suspended Sediment Turbidity
                    let c = state.sediment_c[idx].clamp(0.0, 0.5);
                    let turbidity = (c / 0.08).clamp(0.0, 1.0);
                    let (mud_r, mud_g, mud_b) = (165.0, 115.0, 65.0);
                    water_r = water_r * (1.0 - turbidity) + mud_r * turbidity;
                    water_g = water_g * (1.0 - turbidity) + mud_g * turbidity;
                    water_b = water_b * (1.0 - turbidity) + mud_b * turbidity;

                    // C. 3D Water Surface Normal & Specular Sun Glint
                    let nx = -deta_x * 2.8;
                    let ny = -deta_y * 2.8;
                    let n_len = (nx * nx + ny * ny + 1.0).sqrt();
                    let norm_x = nx / n_len;
                    let norm_y = ny / n_len;
                    let norm_z = 1.0 / n_len;

                    let n_dot_h = (norm_x * (-0.209) + norm_y * (-0.247) + norm_z * 0.946).max(0.0);
                    let spec_sun = if n_dot_h > 0.960 {
                        let t = (n_dot_h - 0.960) / (1.0 - 0.960);
                        t.powi(12) * 1.6
                    } else {
                        0.0
                    };

                    // Fresnel sky reflectance
                    let one_minus_cos = (1.0 - norm_z).max(0.0);
                    let fresnel = 0.04 + 0.96 * one_minus_cos.powi(4);
                    let sky_r = 180.0;
                    let sky_g = 215.0;
                    let sky_b = 248.0;

                    water_r = water_r * (1.0 - fresnel * 0.45) + sky_r * (fresnel * 0.45);
                    water_g = water_g * (1.0 - fresnel * 0.45) + sky_g * (fresnel * 0.45);
                    water_b = water_b * (1.0 - fresnel * 0.45) + sky_b * (fresnel * 0.45);

                    water_r += 255.0 * spec_sun;
                    water_g += 248.0 * spec_sun;
                    water_b += 220.0 * spec_sun;

                    // D. Multi-Source Sea Foam
                    let foam = if is_coastal_waves {
                        let rapids_foam = if speed > 2.2 {
                            ((speed - 2.2) / 2.0).clamp(0.0, 0.85)
                        } else {
                            0.0
                        };

                        let shore_foam = if depth < 0.06 && v < -0.10 {
                            ((-v - 0.10) / 0.40).clamp(0.0, 0.70)
                        } else {
                            0.0
                        };

                        let dz_x = (state.z_bed[row_offset + x_next] - state.z_bed[row_offset + x_prev]) * 0.5;
                        let dz_y = (state.z_bed[y_next_offset + x] - state.z_bed[y_prev_offset + x]) * 0.5;
                        let obstacle_impact = -(u * dz_x + v * dz_y);
                        let obstacle_foam = if obstacle_impact > 0.15 {
                            ((obstacle_impact - 0.15) / 0.50).clamp(0.0, 0.75)
                        } else {
                            0.0
                        };

                        (rapids_foam + shore_foam + obstacle_foam).clamp(0.0, 0.92)
                    } else {
                        0.0
                    };

                    if foam > 0.001 {
                        let foam_r = 248.0;
                        let foam_g = 252.0;
                        let foam_b = 255.0;
                        row_slice[byte_idx] = (water_r * (1.0 - foam) + foam_r * foam).clamp(0.0, 255.0) as u8;
                        row_slice[byte_idx + 1] = (water_g * (1.0 - foam) + foam_g * foam).clamp(0.0, 255.0) as u8;
                        row_slice[byte_idx + 2] = (water_b * (1.0 - foam) + foam_b * foam).clamp(0.0, 255.0) as u8;
                    } else {
                        row_slice[byte_idx] = water_r.clamp(0.0, 255.0) as u8;
                        row_slice[byte_idx + 1] = water_g.clamp(0.0, 255.0) as u8;
                        row_slice[byte_idx + 2] = water_b.clamp(0.0, 255.0) as u8;
                    }
                    row_slice[byte_idx + 3] = 255;
                } else {
                    // --- DRY / EXPOSED LAND PASS ---
                    let mut final_r = r_land;
                    let mut final_g = g_land;
                    let mut final_b = b_land;

                    // Wet Sand Specular Gloss ("Mirror Beach")
                    if sat > 0.40 {
                        let dz_x = (state.z_bed[row_offset + x_next] - state.z_bed[row_offset + x_prev]) * 0.5;
                        let dz_y = (state.z_bed[y_next_offset + x] - state.z_bed[y_prev_offset + x]) * 0.5;
                        let land_nx = -dz_x * 2.0;
                        let land_ny = -dz_y * 2.0;
                        let land_len = (land_nx * land_nx + land_ny * land_ny + 1.0).sqrt();
                        let land_dot_h = ((land_nx / land_len) * (-0.209) + (land_ny / land_len) * (-0.247) + (1.0 / land_len) * 0.946).max(0.0);
                        let wet_spec = land_dot_h.powi(22) * (sat - 0.40) * 2.2;

                        final_r = (final_r + 210.0 * wet_spec).min(255.0);
                        final_g = (final_g + 225.0 * wet_spec).min(255.0);
                        final_b = (final_b + 245.0 * wet_spec).min(255.0);
                    }

                    row_slice[byte_idx] = final_r as u8;
                    row_slice[byte_idx + 1] = final_g as u8;
                    row_slice[byte_idx + 2] = final_b as u8;
                    row_slice[byte_idx + 3] = 255;
                }
            }
        });
}
