#![no_std]

extern crate alloc;

#[macro_use]
mod util;

pub mod candidates;
pub mod fir_filter;
pub mod harmonics;
pub mod interpolate;
pub mod polyphase_filter;

use self::candidates::CandidateGenerator;
use self::candidates::CandidateSelector;
use self::harmonics::HarmonicParametersEstimator;

use alloc::collections::VecDeque;
use core::ops::RangeInclusive;

pub struct Irapt {
    parameters:          Parameters,
    estimator:           HarmonicParametersEstimator,
    candidate_generator: CandidateGenerator,
    candidate_selector:  CandidateSelector,
}

pub struct Parameters {
    pub sample_rate: f64,

    pub harmonics_estimation_interval:        f64,
    pub harmonics_estimation_window_duration: f64,
    pub candidate_selection_window_duration:  f64,

    pub pitch_range: RangeInclusive<f64>,

    pub candidate_generator_fft_len:   usize,
    pub half_interpolation_window_len: u32,
    pub interpolation_factor:          u8,

    pub candidate_step_decay: f64,
    pub candidate_max_jump:   usize,
}

pub struct EstimatedPitch {
    pub frequency: f64,
    pub energy:    f64,
}

impl Irapt {
    pub fn new(parameters: Parameters) -> Self {
        let estimator = HarmonicParametersEstimator::new(parameters.harmonics_window_len());

        let candidate_generator = CandidateGenerator::new(
            parameters.candidate_generator_fft_len,
            parameters.half_interpolation_window_len,
            parameters.interpolation_factor,
            parameters.sample_rate,
            parameters.pitch_range.clone(),
        );
        let candidate_selector = CandidateSelector::new(parameters.candidate_selection_window_len(), candidate_generator.window_len());
        Self {
            parameters,
            estimator,
            candidate_generator,
            candidate_selector,
        }
    }

    pub fn process<S: Into<f64> + Copy>(&mut self, samples: &mut VecDeque<S>) -> Option<EstimatedPitch> {
        let step_len = self.parameters.step_len();
        while let Some(harmonics) = self.estimator.process_step(samples, step_len, self.parameters.sample_rate) {
            let mut energy = 0.0;
            let harmonics = harmonics.inspect(|harmonic| {
                energy += harmonic.amplitude * harmonic.amplitude;
            });

            let candidates = self.candidate_generator.process_step(harmonics, self.parameters.sample_rate);

            let mut min_candidate = 0.0;
            let candidates = candidates.inspect(|candidate| {
                min_candidate = candidate.min(min_candidate);
            });

            if let Some(best_candidate_index) = self.candidate_selector.process_step(
                candidates,
                self.parameters.candidate_selection_window_len(),
                self.parameters.candidate_max_jump,
                self.parameters.candidate_step_decay,
            ) {
                let candidate_frequency = self
                    .candidate_generator
                    .candidate_frequency(best_candidate_index, self.parameters.sample_rate);
                return Some(EstimatedPitch {
                    frequency: candidate_frequency,
                    energy:    (-min_candidate / energy.max(1e-4)),
                });
            }
        }
        None
    }
}

//
// Parameters impls
//

impl Parameters {
    const DEFAULT: Self = Self {
        sample_rate: 6000.0,

        harmonics_estimation_interval:        0.005,
        harmonics_estimation_window_duration: 0.05,
        candidate_selection_window_duration:  0.3,

        pitch_range: 50.0..=450.0,

        candidate_generator_fft_len:   16384,
        half_interpolation_window_len: 12,
        interpolation_factor:          2,

        candidate_step_decay: 0.95,
        candidate_max_jump:   23,
    };

    fn candidate_selection_window_len(&self) -> usize {
        (self.candidate_selection_window_duration / self.harmonics_estimation_interval + 0.5) as usize
    }

    fn harmonics_window_len(&self) -> u32 {
        (self.harmonics_estimation_window_duration * self.sample_rate / 2.0).round() as u32 * 2 + 1
    }

    fn step_len(&self) -> usize {
        (self.harmonics_estimation_interval * self.sample_rate).round() as usize
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self::DEFAULT
    }
}
