use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::anyhow;
use plotters::{prelude::*, style::RelativeSize};
use plotters_piston::PistonBackend;
use piston_window::{Event, Loop, PistonWindow, WindowSettings, EventLoop};

pub fn run(display: Arc<DisplayData>) -> anyhow::Result<()> {
    let mut window = WindowSettings::new("IRAPT Test UI", [512; 2])
        .exit_on_esc(true)
        .build::<PistonWindow>()
        .map_err(|error| anyhow!("error creating window: {}", error))?;

    let max_pitches = 60;
    let max_pitch = 500.0;
    let mut pitches = VecDeque::with_capacity(max_pitches);
    let mut tentative_pitches = Vec::new();
    while let Some(event) = window.next() {
        match &event {
            Event::Loop(Loop::Update(_update_args)) => {
                if let Some(new_pitches) = display.pitches.take() {
                    tentative_pitches = Vec::from(new_pitches);
                    if let Some(new_pitch) = tentative_pitches.pop() {
                        if pitches.len() == max_pitches {
                            pitches.pop_front();
                        }
                        pitches.push_back(new_pitch);
                    }
                }
            }
            Event::Loop(Loop::Render(render_args)) => {
                let draw_result: Option<anyhow::Result<()>> = window.draw_2d(&event, |context, graphics, _device| {
                    let [window_width, _window_height] = render_args.window_size;
                    let [draw_width, draw_height] = render_args.draw_size;
                    let backend = PistonBackend::new(
                        (draw_width, draw_height),
                        window_width / draw_width as f64,
                        context,
                        graphics,
                    );

                    let root = backend.into_drawing_area();
                    root.fill(&WHITE)?;

                    let (pitch_chart_area, tentative_pitch_chart_area) = root.split_horizontally(RelativeSize::Width(0.8));

                    let mut pitch_chart = ChartBuilder::on(&pitch_chart_area)
                        .margin(0)
                        .x_label_area_size(0)
                        .y_label_area_size(0)
                        .build_cartesian_2d(0..max_pitches - 1, 0.0..max_pitch)?;

                    pitch_chart
                        .configure_mesh()
                        .disable_axes()
                        .disable_x_mesh()
                        .draw()?;

                    let pitches_offset = max_pitches - pitches.len();
                    pitch_chart.draw_series(LineSeries::new({
                        pitches
                            .iter()
                            .enumerate()
                            .flat_map(|(index, pitch)| pitch.frequency.map(|frequency| (pitches_offset + index, frequency)))
                    }, &RED))?;

                    let mut tentative_pitch_chart = ChartBuilder::on(&tentative_pitch_chart_area)
                        .margin(0)
                        .x_label_area_size(0)
                        .y_label_area_size(0)
                        .build_cartesian_2d(0..tentative_pitches.len(), 0.0..max_pitch)?;

                    tentative_pitch_chart
                        .configure_mesh()
                        .disable_axes()
                        .disable_x_mesh()
                        .draw()?;

                    tentative_pitch_chart.draw_series(LineSeries::new({
                        tentative_pitches
                            .iter()
                            .rev()
                            .enumerate()
                            .flat_map(|(index, pitch)| pitch.frequency.map(|frequency| (index, frequency)))
                    }, &RED))?;

                    Ok(())
                });
                if let Some(draw_result) = draw_result {
                    draw_result?;
                }
            }
            Event::Loop(Loop::AfterRender(_after_render_args)) => (),
            Event::Loop(Loop::Idle(_idle_args)) => (),
            Event::Input(_input, _ts) => (),
            Event::Custom(_id, _data, _ts) => (),
        }
    }
    Ok(())
}
