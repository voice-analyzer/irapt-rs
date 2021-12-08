use std::cmp::Ordering;

use anyhow::Context as _;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, Stream, SupportedStreamConfig, SampleFormat};
use crossbeam_channel::RecvError;

pub struct AudioInput {
    device: Device,
    config: SupportedStreamConfig,
}

pub struct AudioInputJoinHandle {
    error_tx: crossbeam_channel::Sender<AudioInputResult>,
    error_rx: crossbeam_channel::Receiver<AudioInputResult>,
    _stream: Stream,
}

#[derive(Clone)]
pub struct AudioInputJoinCancellationGuard {
    error_tx: crossbeam_channel::Sender<AudioInputResult>,
}

enum AudioInputResult {
    JoinCancelled,
    Err(anyhow::Error),
}

impl AudioInput {
    pub fn new(sample_rate: SampleRate) -> anyhow::Result<Self> {
        let device = cpal::default_host().default_input_device().context("no input device available")?;
        let config_range = device
            .supported_input_configs()
            .context("error querying input device configurations")?
            .filter(|config_range| config_range.sample_format() == SampleFormat::F32)
            .max_by(|a, b| {
                Ordering::Equal
                    .then_with(|| (a.min_sample_rate() <= sample_rate).cmp(&(b.min_sample_rate() <= sample_rate)))
                    .then_with(|| match a.min_sample_rate() {
                        a_min_sample_rate if a_min_sample_rate <= sample_rate => Ordering::Equal,
                        a_min_sample_rate => a_min_sample_rate.cmp(&b.min_sample_rate()).reverse()
                    })
                    .then_with(|| (a.channels() == 1).cmp(&(b.channels() == 1)))
                    .then_with(|| a.channels().cmp(&b.channels()).reverse())
            })
            .context("input device reported no supported configurations available")?;
        let sample_rate = sample_rate.clamp(config_range.min_sample_rate(), config_range.max_sample_rate());
        let config = config_range.with_sample_rate(sample_rate);
        Ok(Self { device, config })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &SupportedStreamConfig {
        &self.config
    }

    pub fn start<F>(&mut self, mut callback: F) -> anyhow::Result<AudioInputJoinHandle>
    where
        F: FnMut(&[f32]) + Send + 'static,
    {
        let (error_tx, error_rx) = crossbeam_channel::bounded(0);
        let error_tx_2 = error_tx.clone();
        let stream = self
            .device
            .build_input_stream(
                &self.config.config(),
                move |samples: &[f32], _info| callback(samples),
                move |error| drop(error_tx_2.send(AudioInputResult::Err(error.into()))),
            )
            .context("error building audio input stream")?;

        stream.play().context("error starting audio input stream")?;

        Ok(AudioInputJoinHandle {
            error_tx,
            error_rx,
            _stream: stream,
        })
    }
}

impl AudioInputJoinHandle {
    pub fn cancellation_guard(&self) -> AudioInputJoinCancellationGuard {
        AudioInputJoinCancellationGuard {
            error_tx: self.error_tx.clone(),
        }
    }

    pub fn join(self) -> anyhow::Result<()> {
        let Self { error_rx, .. } = self;
        match error_rx.recv() {
            Err(RecvError) | Ok(AudioInputResult::JoinCancelled) => Ok(()),
            Ok(AudioInputResult::Err(error)) => Err(error),
        }
    }
}

impl Drop for AudioInputJoinCancellationGuard {
    fn drop(&mut self) {
        let _ = self.error_tx.send(AudioInputResult::JoinCancelled);
    }
}
