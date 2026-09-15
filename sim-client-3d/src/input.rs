use glam::Vec2;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use sim_core::domain::SimDomainDescriptor;
use crate::camera::{Camera, CameraMode};
use crate::passes::{ActiveTool, UiState};

pub struct InputState {
    pub cursor_screen_pos: Vec2,
    pub cursor_in_window: bool,
    pub is_right_down: bool,
    pub is_left_down: bool,
    pub is_shift_down: bool,
    pub is_ctrl_down: bool,
    pub last_mouse_pos: Vec2,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            cursor_screen_pos: Vec2::ZERO,
            cursor_in_window: false,
            is_right_down: false,
            is_left_down: false,
            is_shift_down: false,
            is_ctrl_down: false,
            last_mouse_pos: Vec2::ZERO,
        }
    }

    pub fn handle_event(
        &mut self,
        event: &WindowEvent,
        consumed: bool,
        camera: &mut Camera,
        ui_state: &mut UiState,
        desc: &SimDomainDescriptor,
    ) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_screen_pos = Vec2::new(position.x as f32, position.y as f32);
                self.cursor_in_window = true;

                if !consumed {
                    let delta = self.cursor_screen_pos - self.last_mouse_pos;
                    if self.is_right_down {
                        if self.is_shift_down {
                            camera.pan(delta.x, delta.y);
                        } else {
                            camera.orbit(-delta.x * 0.006, delta.y * 0.006);
                        }
                    }
                }
                self.last_mouse_pos = self.cursor_screen_pos;
            }

            WindowEvent::CursorLeft { .. } => {
                self.cursor_in_window = false;
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if !consumed {
                    let scroll_amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.05,
                    };
                    camera.zoom(scroll_amount);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if !consumed {
                            self.is_left_down = pressed;
                        } else if !pressed {
                            self.is_left_down = false;
                        }
                    }
                    MouseButton::Right => {
                        self.is_right_down = pressed;
                    }
                    _ => {}
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;
                match logical_key {
                    Key::Named(NamedKey::Shift) => self.is_shift_down = pressed,
                    Key::Named(NamedKey::Control) => self.is_ctrl_down = pressed,
                    Key::Named(NamedKey::Tab) if pressed => {
                        camera.toggle_mode();
                        ui_state.camera_mode_name = match camera.mode {
                            CameraMode::Perspective => "Perspective (3D Orbit)",
                            CameraMode::Isometric => "2.5D True Isometric",
                        };
                    }
                    Key::Named(NamedKey::Space) if pressed => {
                        ui_state.is_paused = !ui_state.is_paused;
                    }
                    Key::Character(c) if pressed => match c.as_str() {
                        "r" | "R" => camera.reset(desc.extent_x, desc.extent_y),
                        "1" => ui_state.active_tool = ActiveTool::Water,
                        "2" => ui_state.active_tool = ActiveTool::SandDam,
                        "3" => ui_state.active_tool = ActiveTool::StoneBreakwater,
                        "4" => ui_state.active_tool = ActiveTool::Dig,
                        _ => {}
                    },
                    _ => {}
                }
            }

            _ => {}
        }
    }
}
