use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use core::ops::RangeInclusive;

use num::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::InvalidParameterError;
use crate::harmonics::HarmonicParameter;
use crate::interpolate::InterpolationFilter;

pub struct CandidateGenerator {
    ifft:         Arc<dyn Fft<f64>>,
    ifft_buffer:  Box<[Complex<f64>]>,
    ifft_scratch: Box<[Complex<f64>]>,
    pitch_range:  RangeInclusive<usize>,
    interpolator: InterpolationFilter,
    window_len:   usize,
}

impl CandidateGenerator {
    pub fn new(
        fft_len: usize,
        half_interpolation_window_len: u32,
        interpolation_factor: u8,
        sample_rate: f64,
        pitch_range: RangeInclusive<f64>,
    ) -> Result<Self, InvalidParameterError> {
        let min_pitch_index = (sample_rate / pitch_range.end()).floor() as usize;
        let max_pitch_index = (sample_rate / pitch_range.start()).ceil() as usize;

        let max_half_interpolation_window_len = usize::min(min_pitch_index, fft_len - max_pitch_index) as u32;
        if half_interpolation_window_len > max_half_interpolation_window_len {
            return Err(InvalidParameterError::InterpolationWindowTooLong {
                max_length: max_half_interpolation_window_len,
            });
        }

        let ifft = FftPlanner::new().plan_fft_inverse(fft_len);
        let ifft_buffer = vec![<_>::default(); ifft.len()].into();
        let ifft_scratch = vec![<_>::default(); ifft.get_inplace_scratch_len()].into();
        let interpolator = InterpolationFilter::new(half_interpolation_window_len, interpolation_factor);
        let window_len = (max_pitch_index - min_pitch_index) * usize::from(interpolation_factor) + 1;
        Ok(Self {
            ifft,
            ifft_buffer,
            ifft_scratch,
            pitch_range: min_pitch_index..=max_pitch_index,
            interpolator,
            window_len,
        })
    }

    pub fn candidate_frequency(&self, index: usize, sample_rate: f64) -> f64 {
        let pitch_range = self.pitch_range.end() - self.pitch_range.start();
        let candidate_spacing = pitch_range as f64 / (self.window_len() - 1) as f64;
        sample_rate / (*self.pitch_range.start() as f64 + index as f64 * candidate_spacing)
    }

    pub fn window_len(&self) -> usize {
        self.window_len
    }

    pub fn process_step(&mut self, harmonics: impl Iterator<Item = HarmonicParameter>, sample_rate: f64) -> impl Iterator<Item = f64> + '_ {
        let ifft_order = self.ifft.len() as f64;
        let ifft_ratio = sample_rate / ifft_order;
        self.ifft_buffer.fill(<_>::default());
        for harmonic in harmonics.filter(|harmonic| (0.0..=sample_rate / 2.0).contains(&harmonic.frequency)) {
            // place at the closest corresponding positions of ±frequency in ifft_buffer
            let ifft_buffer_index = (harmonic.frequency / ifft_ratio).round() as usize;
            let normalized_amplitude = harmonic.amplitude * harmonic.amplitude;
            self.ifft_buffer[ifft_buffer_index] = normalized_amplitude.into();
            if ifft_buffer_index != 0 {
                self.ifft_buffer[self.ifft_buffer.len() - ifft_buffer_index] = normalized_amplitude.into();
            }
        }

        self.ifft.process_with_scratch(&mut self.ifft_buffer, &mut self.ifft_scratch);

        let half_interpolation_window_len = self.interpolator.window_len() / 2;
        let pitches = self.ifft_buffer
            [self.pitch_range.start() - half_interpolation_window_len..=self.pitch_range.end() + half_interpolation_window_len]
            .iter()
            .map(move |pitch| pitch.re / 2.0);

        let interpolated = self.interpolator.interpolate(pitches);
        interpolated.map(|candidate| -candidate)
    }
}

#[cfg(test)]
pub mod test;
