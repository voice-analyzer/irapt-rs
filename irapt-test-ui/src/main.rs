use std::collections::VecDeque;
use std::convert::Infallible;
use std::io::Write;
use std::panic::resume_unwind;
use std::sync::Arc;

use anyhow::anyhow;
use anyhow::Context as _;
use cpal::SampleRate;
use cpal::traits::DeviceTrait;
use crossbeam_utils::sync::Parker;
use crossbeam_utils::thread::scope;
use irapt::Irapt;
use irapt_test_ui::input::AudioInput;
use irapt_test_ui::ui;
use irapt_test_ui::ui::DisplayData;
use log::LevelFilter;
use rubato::Resampler as _;
use scopeguard::defer;

type Resampler = rubato::FftFixedOut<f32>;

const RESAMPLER_CHUNK_SIZE: usize = 60;

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .format(|buf, record| {
            writeln!(
                buf,
                "{timestamp} {level:<5} [{target}] {module_path}:{line} {delimiter} {message}",
                timestamp = buf.timestamp_millis(),
                level = buf.default_styled_level(record.level()),
                target = record.target(),
                module_path = record.module_path().unwrap_or("<none>"),
                line = record.line().unwrap_or(0),
                message = record.args(),
                delimiter = buf.style().set_color(env_logger::fmt::Color::Black).set_intense(true).value("==="),
            )
        })
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .init();

    let display = Arc::new(DisplayData::default());
    let display_weak = Arc::downgrade(&display);

    let parameters = irapt::Parameters::default();
    let downsampled_rate = parameters.sample_rate;

    let mut input = AudioInput::new(SampleRate(downsampled_rate as u32)).context("error opening audio input")?;

    log::info!(
        "selected default input device \"{device_name}\" at sample rate {sample_rate} with {channel_count} channels",
        device_name = input.device().name()?,
        sample_rate = input.config().sample_rate().0,
        channel_count = input.config().channels(),
    );

    let sample_rate = input.config().sample_rate().0;
    let mut resampler = Resampler::new(sample_rate as usize, downsampled_rate as usize, RESAMPLER_CHUNK_SIZE, 1, 1);
    let mut irapt = Irapt::new(parameters).map_err(|error| anyhow!("error initializing irapt: {:?}", error))?;

    let (mut input_samples_tx, mut input_samples_rx) = ringbuf::RingBuffer::new(sample_rate as usize / 2).split();
    let input_samples_parker = Parker::new();
    let input_samples_unparker = input_samples_parker.unparker().clone();

    let (input_cancellation_guard_tx, input_cancellation_guard_rx) = crossbeam_channel::bounded(0);

    let scope_result = scope(|scope| -> anyhow::Result<()> {
        scope.spawn(|_scope| -> anyhow::Result<()> {
            let input_samples_unparker = input_samples_unparker.clone();
            let input_handle = input.start(move |samples| {
                input_samples_tx.push_slice(samples);
                input_samples_unparker.unpark();
            })?;

            input_cancellation_guard_tx
                .send(input_handle.cancellation_guard())
                .context("main thread stopped")?;

            input_handle.join().context("audio input error")?;
            Ok(())
        });

        scope.spawn(|_scope| -> Option<Infallible> {
            let input_samples_parker = input_samples_parker;
            let mut unresampled = Vec::<f32>::with_capacity(sample_rate as usize / 2);
            let mut downsampled = VecDeque::<f32>::with_capacity(downsampled_rate as usize / 2);
            loop {
                input_samples_parker.park();

                let display = display_weak.upgrade()?;

                let mut new_sample_count = 0;
                input_samples_rx.access(|new_samples_head, new_samples_tail| {
                    new_sample_count = new_samples_head.len() + new_samples_tail.len();

                    for mut new_samples in [new_samples_head, new_samples_tail] {
                        while let Some(new_samples_chunk) = new_samples.get(..resampler.nbr_frames_needed() - unresampled.len()) {
                            new_samples = &new_samples[new_samples_chunk.len()..];
                            if unresampled.is_empty() {
                                downsampled.extend(resampler.process(&[new_samples_chunk]).unwrap().pop().unwrap());
                            } else {
                                unresampled.extend(new_samples_chunk);
                                downsampled.extend(resampler.process(&[&unresampled]).unwrap().pop().unwrap());
                                unresampled.clear();
                            }
                        }
                        unresampled.extend(new_samples);
                    }
                });
                input_samples_rx.discard(new_sample_count);

                while let Some(output) = irapt.process(&mut downsampled) {
                    let pitches = output.pitch_estimates().collect();
                    display.set_pitches(pitches);
                }
            }
        });

        let _input_cancellation_guard = input_cancellation_guard_rx.recv().context("audio input thread died")?;
        defer! { input_samples_unparker.unpark(); }

        ui::run(display).context("ui error")?;
        Ok(())
    });
    let ui_result = scope_result.unwrap_or_else(|error| resume_unwind(error));
    ui_result
}
