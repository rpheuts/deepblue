use macroquad::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::backend::SimulationBackend;
use crate::camera::CameraState;

pub fn handle_camera_input(camera: &mut CameraState, prev_mouse_pos: &mut (f32, f32)) {
    let (mx, my) = mouse_position();
    let (_, mouse_wheel_y) = mouse_wheel();

    if mouse_wheel_y.abs() > 0.01 {
        let zoom_factor = if mouse_wheel_y > 0.0 { 1.15 } else { 0.87 };
        let new_zoom = (camera.zoom * zoom_factor).clamp(1.0, 8.0);
        let screen_w = screen_width();
        let screen_h = screen_height();
        let norm_mx = (mx / screen_w).clamp(0.0, 1.0);
        let norm_my = (my / screen_h).clamp(0.0, 1.0);

        let old_w = 1.0 / camera.zoom;
        let old_h = 1.0 / camera.zoom;
        let new_w = 1.0 / new_zoom;
        let new_h = 1.0 / new_zoom;

        camera.target_x += (norm_mx - 0.5) * (old_w - new_w);
        camera.target_y += (norm_my - 0.5) * (old_h - new_h);
        camera.zoom = new_zoom;
        camera.target_x = camera.target_x.clamp(0.0, 1.0);
        camera.target_y = camera.target_y.clamp(0.0, 1.0);
    }

    if is_mouse_button_down(MouseButton::Middle) {
        let dx = mx - prev_mouse_pos.0;
        let dy = my - prev_mouse_pos.1;
        let pan_speed_x = (dx / screen_width()) / camera.zoom;
        let pan_speed_y = (dy / screen_height()) / camera.zoom;
        camera.target_x = (camera.target_x - pan_speed_x).clamp(0.0, 1.0);
        camera.target_y = (camera.target_y - pan_speed_y).clamp(0.0, 1.0);
    }

    let pan_step = 0.008 / camera.zoom;
    if is_key_down(KeyCode::Left) { camera.target_x = (camera.target_x - pan_step).max(0.0); }
    if is_key_down(KeyCode::Right) { camera.target_x = (camera.target_x + pan_step).min(1.0); }
    if is_key_down(KeyCode::Up) { camera.target_y = (camera.target_y - pan_step).max(0.0); }
    if is_key_down(KeyCode::Down) { camera.target_y = (camera.target_y + pan_step).min(1.0); }

    if is_key_pressed(KeyCode::C) || is_key_pressed(KeyCode::Key0) {
        camera.reset();
    }

    *prev_mouse_pos = (mx, my);
}

pub fn handle_tools(
    sim: &mut Box<dyn SimulationBackend>,
    camera: &CameraState,
    desc: &SimDomainDescriptor,
    bed_dirty: &mut bool,
) {
    let (mx, my) = mouse_position();
    let mouse_in_window = mx >= 0.0 && mx <= screen_width() && my >= 0.0 && my <= screen_height();

    if mouse_in_window && !is_mouse_button_down(MouseButton::Middle) {
        let (gx, gy) = camera.screen_to_grid(mx, my, screen_width(), screen_height(), desc.grid_res_x as f32, desc.grid_res_y as f32);
        let base_radius = 10.0 * (desc.grid_res_x as f32 / 512.0);
        let radius = (base_radius / camera.zoom.sqrt()).max(3.0) as i32;
        let r2 = radius * radius;

        let ctrl_down = is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl);
        let shift_down = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);

        let add_water = is_mouse_button_down(MouseButton::Left) && !shift_down && !ctrl_down;
        let build_sand_dam = shift_down && is_mouse_button_down(MouseButton::Left);
        let dig_trench = is_mouse_button_down(MouseButton::Right) && !shift_down && !ctrl_down;
        let dump_sand = shift_down && is_mouse_button_down(MouseButton::Right);
        let place_stone_wall = ctrl_down && is_mouse_button_down(MouseButton::Left);
        let demolish_stone = ctrl_down && is_mouse_button_down(MouseButton::Right);

        if add_water || build_sand_dam || dig_trench || dump_sand || place_stone_wall || demolish_stone {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= r2 {
                        let px = (gx + dx).clamp(1, desc.grid_res_x as i32 - 2) as u32;
                        let py = (gy + dy).clamp(1, desc.grid_res_y as i32 - 2) as u32;
                        let idx = sim.current_state().idx(px, py);

                        if add_water {
                            sim.current_state_mut().h[idx] += 0.3;
                        } else if build_sand_dam {
                            sim.current_state_mut().z_bed[idx] += 0.12;
                        } else if dump_sand {
                            sim.current_state_mut().z_bed[idx] += 0.08;
                            sim.current_state_mut().soil_sat[idx] = 0.10;
                        } else if dig_trench {
                            let cur_z = sim.current_state().z_bed[idx];
                            let bedrock = sim.current_state().bedrock_z[idx];
                            sim.current_state_mut().z_bed[idx] = (cur_z - 0.12).max(bedrock);
                        } else if place_stone_wall {
                            sim.current_state_mut().z_bed[idx] += 0.15;
                            sim.current_state_mut().bedrock_z[idx] = sim.current_state().z_bed[idx];
                        } else if demolish_stone {
                            let cur_z = sim.current_state().z_bed[idx];
                            sim.current_state_mut().z_bed[idx] = (cur_z - 0.15).max(-1.0);
                            sim.current_state_mut().bedrock_z[idx] = (sim.current_state().z_bed[idx] - 0.45).max(-1.5);
                        }
                    }
                }
            }
            if add_water && !build_sand_dam && !dig_trench && !dump_sand && !place_stone_wall && !demolish_stone {
                sim.upload_water_depth();
            } else {
                sim.upload_state();
                *bed_dirty = true;
            }
        }
    }
}
