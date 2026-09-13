use macroquad::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::DoubleBufferedGrid;
use sim_backend::WgpuSimulator;

#[macroquad::main("DeepBlue 2D Vis (GPU Compute)")]
async fn main() {
    let mut desc = SimDomainDescriptor::default();
    desc.grid_res_x = 512;
    desc.grid_res_y = 512;

    let mut grid = DoubleBufferedGrid::new(desc);

    // Initial state: A drop of water in the middle
    let center_x = desc.grid_res_x / 2;
    let center_y = desc.grid_res_y / 2;
    let radius = 40;

    for y in 0..desc.grid_res_y {
        for x in 0..desc.grid_res_x {
            let idx = grid.current.idx(x, y);
            let dx = x as f32 - center_x as f32;
            let dy = y as f32 - center_y as f32;
            if dx * dx + dy * dy < (radius * radius) as f32 {
                grid.current.h[idx] = 10.0;
            }
        }
    }

    let mut gpu_sim = WgpuSimulator::new(grid).await;
    let dt = 0.008; // Safe CFL timestep

    // Fast texture blitting buffer
    let mut img = Image::gen_image_color(desc.grid_res_x as u16, desc.grid_res_y as u16, BLACK);
    let texture = Texture2D::from_image(&img);
    texture.set_filter(FilterMode::Linear);

    loop {
        // Step the simulation
        for _ in 0..4 {
            gpu_sim.step(dt);
        }

        // Sync data back to CPU for visualization
        gpu_sim.sync_to_cpu().await;

        clear_background(BLACK);

        // Blit fluid depth into texture pixel buffer
        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = gpu_sim.cpu_grid.current.idx(x, y);
                let h = gpu_sim.cpu_grid.current.h[idx];
                let byte_idx = (idx * 4) as usize;

                if h > 0.01 {
                    let intensity = (h / 10.0).clamp(0.2, 1.0);
                    img.bytes[byte_idx + 0] = 0;
                    img.bytes[byte_idx + 1] = (128.0 * intensity) as u8;
                    img.bytes[byte_idx + 2] = (255.0 * intensity) as u8;
                    img.bytes[byte_idx + 3] = 255;
                } else {
                    img.bytes[byte_idx + 0] = 15;
                    img.bytes[byte_idx + 1] = 15;
                    img.bytes[byte_idx + 2] = 20;
                    img.bytes[byte_idx + 3] = 255;
                }
            }
        }

        texture.update(&img);

        draw_texture_ex(
            &texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                ..Default::default()
            },
        );

        draw_text(&format!("FPS: {}", get_fps()), 10.0, 30.0, 24.0, WHITE);
        draw_text(
            &format!("Grid: {}x{}", desc.grid_res_x, desc.grid_res_y),
            10.0,
            60.0,
            20.0,
            LIGHTGRAY,
        );

        next_frame().await
    }
}
