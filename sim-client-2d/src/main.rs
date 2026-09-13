use macroquad::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::DoubleBufferedGrid;
use sim_backend::WgpuSimulator;

#[macroquad::main("DeepBlue 2D Vis (GPU Compute)")]
async fn main() {
    let mut desc = SimDomainDescriptor::default();
    // With GPU we can easily do 512x512 even on laptops!
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
            if dx*dx + dy*dy < (radius*radius) as f32 {
                grid.current.h[idx] = 10.0;
            }
        }
    }

    let mut gpu_sim = WgpuSimulator::new(grid).await;
    let dt = 0.016; // Fixed timestep

    loop {
        // Step the simulation 5 times per frame for faster fluid motion
        for _ in 0..5 {
            gpu_sim.step(dt);
        }
        
        // Sync data back to CPU to render it in macroquad
        gpu_sim.sync_to_cpu().await;

        clear_background(BLACK);

        let cell_w = screen_width() / desc.grid_res_x as f32;
        let cell_h = screen_height() / desc.grid_res_y as f32;

        // Draw the water depth
        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = gpu_sim.cpu_grid.current.idx(x, y);
                let h = gpu_sim.cpu_grid.current.h[idx];
                
                if h > 0.01 {
                    // Map depth to a blue intensity
                    let intensity = (h / 10.0).clamp(0.2, 1.0);
                    let color = Color::new(0.0, 0.5 * intensity, intensity, 1.0);
                    draw_rectangle(
                        x as f32 * cell_w,
                        y as f32 * cell_h,
                        cell_w,
                        cell_h,
                        color,
                    );
                }
            }
        }

        draw_text(&format!("FPS: {}", get_fps()), 10.0, 30.0, 20.0, WHITE);

        next_frame().await
    }
}
