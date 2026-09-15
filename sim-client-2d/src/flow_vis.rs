use macroquad::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::GridState;
use crate::camera::CameraState;
use crate::types::FastRng;

pub struct FlowParticle {
    pub x: f32,
    pub y: f32,
    pub prev_x: f32,
    pub prev_y: f32,
    pub life: f32,
    pub max_life: f32,
    pub speed: f32,
}

pub fn respawn_particle(
    p: &mut FlowParticle,
    state: &GridState,
    rng: &mut FastRng,
    width: u32,
    height: u32,
) {
    p.life = rng.range_f32(1.0, 3.5);
    p.max_life = p.life;
    p.speed = 0.0;

    // Sample randomly to find active flowing water
    for _ in 0..20 {
        let gx = rng.range_f32(2.0, (width - 3) as f32) as u32;
        let gy = rng.range_f32(2.0, (height - 3) as f32) as u32;
        let idx = state.idx(gx, gy);
        if state.h[idx] > 0.03 {
            p.x = gx as f32 + rng.next_f32();
            p.y = gy as f32 + rng.next_f32();
            p.prev_x = p.x;
            p.prev_y = p.y;
            return;
        }
    }

    // Fallback: spawn in lower or middle domain
    p.x = rng.range_f32(10.0, (width - 10) as f32);
    p.y = rng.range_f32(height as f32 * 0.35, height as f32 * 0.85);
    p.prev_x = p.x;
    p.prev_y = p.y;
}

pub fn render_flow_vectors(
    state: &GridState,
    camera: &CameraState,
    desc: &SimDomainDescriptor,
    screen_w: f32,
    screen_h: f32,
) {
    let step: usize = (20.0 / camera.zoom.sqrt()).max(10.0) as usize;
    for gy in (step / 2..desc.grid_res_y as usize).step_by(step) {
        for gx in (step / 2..desc.grid_res_x as usize).step_by(step) {
            let idx = state.idx(gx as u32, gy as u32);
            let depth = state.h[idx];
            if depth > 0.02 {
                let u = state.u[idx];
                let v = state.v[idx];
                let speed_sq = u * u + v * v;
                if speed_sq > 0.0064 {
                    let speed = speed_sq.sqrt();
                    let (sx, sy) = camera.grid_to_screen(
                        gx as f32 + 0.5,
                        gy as f32 + 0.5,
                        screen_w,
                        screen_h,
                        desc.grid_res_x as f32,
                        desc.grid_res_y as f32,
                    );
                    if sx >= -20.0 && sx <= screen_w + 20.0 && sy >= -20.0 && sy <= screen_h + 20.0 {
                        let dir_x = u / speed;
                        let dir_y = v / speed;
                        let len = (speed * 12.0 * camera.zoom.sqrt()).clamp(4.0, 36.0);
                        let ex = sx + dir_x * len;
                        let ey = sy + dir_y * len;
                        let alpha = (speed / 1.5).clamp(0.30, 0.85);

                        draw_line(sx, sy, ex, ey, 1.6, Color::new(0.70, 0.92, 1.0, alpha));
                        draw_circle(ex, ey, 1.3, Color::new(1.0, 1.0, 1.0, alpha * 0.95));
                    }
                }
            }
        }
    }
}

pub fn update_and_render_particles(
    particles: &mut [FlowParticle],
    state: &GridState,
    camera: &CameraState,
    desc: &SimDomainDescriptor,
    rng: &mut FastRng,
    is_paused: bool,
    frame_dt: f32,
    screen_w: f32,
    screen_h: f32,
) {
    for p in particles.iter_mut() {
        if !is_paused {
            p.prev_x = p.x;
            p.prev_y = p.y;

            let gx = p.x.clamp(1.0, (desc.grid_res_x - 2) as f32) as u32;
            let gy = p.y.clamp(1.0, (desc.grid_res_y - 2) as f32) as u32;
            let idx = state.idx(gx, gy);
            let depth = state.h[idx];
            let u = state.u[idx];
            let v = state.v[idx];
            let speed = (u * u + v * v).sqrt();
            p.speed = speed;

            p.life -= frame_dt;
            if p.life <= 0.0
                || depth < 0.02
                || p.x < 2.0
                || p.x >= (desc.grid_res_x - 2) as f32
                || p.y < 2.0
                || p.y >= (desc.grid_res_y - 2) as f32
            {
                respawn_particle(p, state, rng, desc.grid_res_x, desc.grid_res_y);
            } else {
                // Advect with velocity field
                p.x += u * 32.0 * frame_dt;
                p.y += v * 32.0 * frame_dt;
            }
        }

        let (sx0, sy0) = camera.grid_to_screen(
            p.prev_x,
            p.prev_y,
            screen_w,
            screen_h,
            desc.grid_res_x as f32,
            desc.grid_res_y as f32,
        );
        let (sx1, sy1) = camera.grid_to_screen(
            p.x,
            p.y,
            screen_w,
            screen_h,
            desc.grid_res_x as f32,
            desc.grid_res_y as f32,
        );

        if (sx0 >= -20.0 && sx0 <= screen_w + 20.0 && sy0 >= -20.0 && sy0 <= screen_h + 20.0)
            || (sx1 >= -20.0 && sx1 <= screen_w + 20.0 && sy1 >= -20.0 && sy1 <= screen_h + 20.0)
        {
            let dist_sq = (sx1 - sx0).powi(2) + (sy1 - sy0).powi(2);
            if dist_sq < (60.0 * camera.zoom).powi(2) {
                let life_alpha = (p.life / p.max_life).clamp(0.0, 1.0);
                let speed_alpha = (p.speed / 1.0).clamp(0.25, 0.95);
                let alpha = life_alpha * speed_alpha;
                draw_line(
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    (1.8 * camera.zoom.sqrt()).clamp(1.6, 4.0),
                    Color::new(0.88, 0.96, 1.0, alpha),
                );
                if speed_alpha > 0.40 {
                    draw_circle(
                        sx1,
                        sy1,
                        (1.2 * camera.zoom.sqrt()).clamp(1.2, 3.0),
                        Color::new(1.0, 1.0, 1.0, alpha * 0.90),
                    );
                }
            }
        }
    }
}
