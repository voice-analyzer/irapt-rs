use crate::polyphase_filter::PolyphaseFilter;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec;
use core::f64::consts::PI;
use itertools::zip;
use num::Complex;

pub struct HarmonicParametersEstimator {
    filter:          PolyphaseFilter,
    next_step:       usize,
    channel_buffers: (Box<[Complex<f64>]>, Box<[Complex<f64>]>),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HarmonicParameter {
    pub amplitude: f64,
    pub frequency: f64,
}

const SUBCHANNEL_MERGE_WINDOW_LEN: usize = 4;

impl HarmonicParametersEstimator {
    pub fn new(window_len: u32) -> Self {
        let channel_buffer = vec![Complex::new(0.0, 0.0); window_len as usize + 44].into_boxed_slice();
        let filter = PolyphaseFilter::new(window_len, channel_buffer.len());
        Self {
            filter,
            next_step: channel_buffer.len() + 1,
            channel_buffers: (channel_buffer.clone(), channel_buffer),
        }
    }

    pub fn process_step<S: Into<f64> + Copy>(
        &mut self,
        samples: &mut VecDeque<S>,
        step_len: usize,
        sample_rate: f64,
    ) -> Option<impl Iterator<Item = HarmonicParameter> + '_> {
        if samples.len() < self.next_step {
            return None;
        }
        let (head_filter_out, last_filter_out) = &mut self.channel_buffers;
        let step = samples.range(..self.next_step).rev().map(|sample| (*sample).into());
        self.filter.process(step.clone().skip(1), head_filter_out);
        self.filter.process(step, last_filter_out);

        self.next_step += step_len;
        if let Some(drain_len) = self.next_step.checked_sub(head_filter_out.len() * 2) {
            self.next_step -= drain_len;
            samples.drain(..drain_len);
        }

        let head_filter_merged = head_filter_out[..head_filter_out.len() / 2]
            .windows(SUBCHANNEL_MERGE_WINDOW_LEN)
            .map(|window| window.iter().sum::<Complex<f64>>());
        let last_filter_merged = last_filter_out[..last_filter_out.len() / 2]
            .windows(SUBCHANNEL_MERGE_WINDOW_LEN)
            .map(|window| window.iter().sum::<Complex<f64>>());

        let harmonics = zip(head_filter_merged, last_filter_merged).map(move |(head, last)| HarmonicParameter {
            amplitude: last.norm() * 2.0,
            frequency: phase_diff(last.arg(), head.arg()) / (2.0 * PI) * sample_rate,
        });

        Some(harmonics)
    }
}

fn phase_diff(phase: f64, prev_phase: f64) -> f64 {
    phase_unwrap(phase, prev_phase) - prev_phase
}

fn phase_unwrap(phase: f64, prev_phase: f64) -> f64 {
    phase - f64::round((phase - prev_phase) / 2.0 / PI) * 2.0 * PI
}

#[cfg(test)]
pub mod test;
