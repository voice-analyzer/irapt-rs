use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use core::ops::{Range, RangeInclusive};

use num::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::InvalidParameterError;
use crate::harmonics::HarmonicParameter;
use crate::interpolate::InterpolationFilter;

pub struct CandidateGenerator {
    ifft:              Arc<dyn Fft<f64>>,
    ifft_buffer:       Box<[Complex<f64>]>,
    ifft_scratch:      Box<[Complex<f64>]>,
    pitch_index_range: RangeInclusive<usize>,
    interpolator:      InterpolationFilter,
    window_len:        usize,
}

#[derive(Clone)]
pub struct CandidateFrequencyIter {
    sample_rate: f64,
    pitch_range_start: f64,
    candidate_spacing: f64,
    candidate_indices: Range<usize>,
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
            pitch_index_range: min_pitch_index..=max_pitch_index,
            interpolator,
            window_len,
        })
    }

    pub fn candidates_len(&self) -> usize {
        self.window_len
    }

    pub fn candidate_frequencies(&self, sample_rate: f64) -> CandidateFrequencyIter {
        let pitch_range = self.pitch_index_range.end() - self.pitch_index_range.start();
        CandidateFrequencyIter {
            sample_rate,
            candidate_spacing: pitch_range as f64 / (self.window_len - 1) as f64,
            pitch_range_start: *self.pitch_index_range.start() as f64,
            candidate_indices: 0..self.window_len,
        }
    }

    pub fn normalized_candidate_frequencies(
        &self,
        sample_rate: f64,
        pitch_range: RangeInclusive<f64>,
    ) -> impl Iterator<Item = f64> + ExactSizeIterator + '_ {
        let pitch_range_width = pitch_range.end() - pitch_range.start();
        self.candidate_frequencies(sample_rate)
            .map(move |candidate_frequency| (candidate_frequency - pitch_range.start()) / pitch_range_width)
    }

    // process_step is split into two functions to work around overly-constrained lifetimes when returning impl Traits from a function with
    // generic parameters.
    pub fn process_step_harmonics(&mut self, harmonics: impl Iterator<Item = HarmonicParameter>, sample_rate: f64) {
        let ifft_order = self.ifft.len() as f64;
        let ifft_ratio = sample_rate / ifft_order;
        self.ifft_buffer.fill(<_>::default());
        let mut amplitude_square_sum = 0.0;
        for harmonic in harmonics.filter(|harmonic| (0.0..=sample_rate / 2.0).contains(&harmonic.frequency)) {
            // place at the closest corresponding positions of ±frequency in ifft_buffer
            let ifft_buffer_index = (harmonic.frequency / ifft_ratio).round() as usize;
            let amplitude_square = harmonic.amplitude * harmonic.amplitude;
            amplitude_square_sum += amplitude_square;
            self.ifft_buffer[ifft_buffer_index] = amplitude_square.into();
            if ifft_buffer_index != 0 {
                self.ifft_buffer[self.ifft_buffer.len() - ifft_buffer_index] = amplitude_square.into();
            }
        }

        if amplitude_square_sum != 0.0 {
            for ifft_buffer_element in &mut *self.ifft_buffer {
                *ifft_buffer_element /= amplitude_square_sum;
            }
        }

        self.ifft.process_with_scratch(&mut self.ifft_buffer, &mut self.ifft_scratch);
    }

    pub fn generate_step_candidates(&mut self) -> impl Iterator<Item = f64> + '_ {
        let half_interpolation_window_len = self.interpolator.window_len() / 2;
        let pitches = self.ifft_buffer
            [self.pitch_index_range.start() - half_interpolation_window_len..=self.pitch_index_range.end() + half_interpolation_window_len]
            .iter()
            .map(move |pitch| pitch.re / 2.0);

        let interpolated = self.interpolator.interpolate(pitches);
        interpolated.map(|candidate| -candidate)
    }
}

impl Iterator for CandidateFrequencyIter {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sample_rate / (self.pitch_range_start + self.candidate_indices.next()? as f64 * self.candidate_spacing))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.candidate_indices.size_hint()
    }
}

impl ExactSizeIterator for CandidateFrequencyIter {}

impl DoubleEndedIterator for CandidateFrequencyIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.sample_rate / (self.pitch_range_start + self.candidate_indices.next_back()? as f64 * self.candidate_spacing))
    }
}

#[cfg(test)]
pub mod test;
