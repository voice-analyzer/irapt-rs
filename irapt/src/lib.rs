#![no_std]
#![warn(missing_docs)]

//! `irapt` is an implementation of the IRAPT pitch estimation algorithm.
//!
//! IRAPT is an "instantaneous" version of the Robust Algorithm for Pitch Tracking (RAPT).
//!
//! # Usage
//!
//! Currently, the [parameters](Parameters) to [`Irapt`] are technical and may be difficult to tune, but the
//! [`Parameters::default`] provides a sensible set of defaults for ordinary human speech which is computationally
//! efficient, given the input can be resampled to the default [`Parameters::sample_rate`].
//!
//! The input must be given as a [`VecDeque`] to [`Irapt::process`] which is to facilitate the sliding analysis window.
//! The number of samples removed from the buffer by `process` can be calculated on each invocation in order to track
//! the global sample index at which each pitch is estimated:
//!
//! ```
//! use irapt::{Irapt, Parameters};
//! use std::collections::VecDeque;
//! use std::f64::consts::PI;
//!
//! let parameters = Parameters::default();
//! let mut irapt = Irapt::new(parameters.clone()).expect("the default parameters should be valid");
//!
//! let mut sample_buffer = (0..parameters.sample_rate as usize)
//!     .map(|sample_index| f64::sin(sample_index as f64 / parameters.sample_rate * 2.0 * PI * 100.0))
//!     .collect::<VecDeque<_>>();
//!
//! let mut sample_index = 0;
//! while let (initial_sample_buffer_len, Some(estimated_pitch)) = (
//!     sample_buffer.len(),
//!     irapt.process(&mut sample_buffer),
//! ) {
//!     let estimated_pitch_index = sample_index + estimated_pitch.index;
//!     let estimated_pitch_time = estimated_pitch_index as f64 / parameters.sample_rate;
//!     println!("estimated pitch at {:0.3}: {}Hz with energy {}",
//!              estimated_pitch_time, estimated_pitch.frequency, estimated_pitch.energy);
//!     sample_index += initial_sample_buffer_len - sample_buffer.len();
//! }
//! ```

extern crate alloc;

#[macro_use]
mod util;

#[doc(hidden)]
pub mod candidates;
pub mod error;
#[doc(hidden)]
pub mod fir_filter;
#[doc(hidden)]
pub mod harmonics;
#[doc(hidden)]
pub mod interpolate;
#[doc(hidden)]
pub mod polyphase_filter;

use self::candidates::CandidateGenerator;
use self::candidates::CandidateSelector;
use self::error::InvalidParameterError;
use self::harmonics::HarmonicParametersEstimator;

use alloc::collections::VecDeque;
use core::ops::RangeInclusive;

/// Implementation of the IRAPT pitch estimation algorithm.
///
/// IRAPT is an "instantaneous" version of the Robust Algorithm for Pitch Tracking. Though pitch estimates are provided
/// every [`harmonics_estimation_interval`] (`0.005` seconds by default), a larger sliding window of
/// [`candidate_selection_window_duration`] (`0.3` seconds by default) is used to improve accuracy at the cost of a
/// small delay.
///
/// [`harmonics_estimation_interval`]: Parameters::harmonics_estimation_interval
/// [`candidate_selection_window_duration`]: Parameters::candidate_selection_window_duration
pub struct Irapt {
    parameters:          Parameters,
    estimator:           HarmonicParametersEstimator,
    candidate_generator: CandidateGenerator,
    candidate_selector:  CandidateSelector,
}

/// Various tunable parameters for [`Irapt`].
///
/// The [`Default`] implementation provides suggested defaults for all parameters, given that the input is resampled
/// near to the suggested default sample rate.
#[derive(Clone, Debug)]
pub struct Parameters {
    /// The constant sample rate, in Hz, the input was sampled with.
    ///
    /// The suggested default is `6000.0`.
    pub sample_rate: f64,

    /// Interval, in seconds, at which harmonics of the input are estimated.
    ///
    /// The suggested default is `0.005`.
    pub harmonics_estimation_interval: f64,

    /// Duration, in seconds, of the sliding window upon which harmonics of the input are estimated.
    ///
    /// The suggested default is `0.05`.
    pub harmonics_estimation_window_duration: f64,

    /// Duration, in seconds, of the sliding window upon which pitches are estimated.
    ///
    /// A shorter candidate selection window will be more responsive to fluctuations in input, but less accurate. The
    /// suggested default is `0.3`.
    pub candidate_selection_window_duration: f64,

    /// Frequency range, in Hz, within which to detect pitch.
    ///
    /// Wider frequency ranges require a larger [`candidate_generator_fft_len`] or [`sample_rate`] to maintain adequate
    /// frequency resolution of pitch detection. The suggested default for any human speech is `50.0..=450.0`.
    ///
    /// [`candidate_generator_fft_len`]: Self::candidate_generator_fft_len
    /// [`sample_rate`]: Self::sample_rate
    pub pitch_range: RangeInclusive<f64>,

    /// Size of the FFT used for candidate generation.
    ///
    /// The candidate generation FFT size affects the frequency resolution of pitch detection. Larger FFT sizes result
    /// in a higher resolution. The suggested default is `16384`.
    ///
    /// Certain FFT sizes, e.g. powers of two, are more computationally efficient than others. See the
    /// [`rustfft`] crate for the supported optimizations based on FFT size.
    ///
    /// [`rustfft`]: https://docs.rs/rustfft
    pub candidate_generator_fft_len: usize,

    /// Half-length of the window of the interpolator used on generated pitch candidates.
    ///
    /// A window too short for the given [`candidate_generator_fft_len`] will suffer from artifacts resulting from poor
    /// interpolation. The suggested default is `12`.
    ///
    /// The window half-length must be less than or equal to both:
    ///
    ///  * `(sample_rate / pitch_range.end()).floor()`, and
    ///  * `candidate_generator_fft_len - (sample_rate / pitch_range.start()).ceil()`
    ///
    /// [`candidate_generator_fft_len`]: Self::candidate_generator_fft_len
    pub half_interpolation_window_len: u32,

    /// Number of pitch candidates to interpolate in between each generated pitch candidate.
    ///
    /// The suggested default in `2`.
    pub interpolation_factor: u8,

    /// Taper factor applied to candidates within a time step.
    ///
    /// Candidates within a single time step will be weighted from `1.0 - candidate_taper..=1.0` linearly proportional
    /// to their frequencies. The suggested default is `0.25`.
    pub candidate_taper: f64,

    /// Decay factor applied to candidates at each time step within the given [`candidate_selection_window_duration`].
    ///
    /// The suggested default is `0.95`.
    ///
    /// [`candidate_selection_window_duration`]: Self::candidate_selection_window_duration
    pub candidate_step_decay: f64,

    /// Assumed maximum distance a valid pitch will change within the [`harmonics_estimation_interval`].
    ///
    /// The unit of distance is in candidates, which is an arbitrary logarithmic frequency scale.
    ///
    /// [`harmonics_estimation_interval`]: Self::harmonics_estimation_interval
    pub candidate_max_jump: usize,
}

/// Estimate of the current pitch of the input.
pub struct EstimatedPitch {
    /// Frequency, in Hz, of the estimated pitch.
    pub frequency: f64,
    /// Arbitrary measure, from `0.0..=1.0`, of the energy associated with the estimated pitch.
    pub energy:    f64,
    /// The index within the input buffer (_before_ removal of consumed samples) at which this pitch was estimated.
    pub index:     usize,
}

impl Irapt {
    /// Constructs a new `Irapt`.
    ///
    /// # Errors
    ///
    /// If any of the supplied parameters are invalid or conflict with each other, then an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use irapt::{Irapt, Parameters};
    ///
    /// let mut irapt = Irapt::new(Parameters::default()).expect("the default parameters should be valid");
    /// ```
    pub fn new(parameters: Parameters) -> Result<Self, InvalidParameterError> {
        let estimator = HarmonicParametersEstimator::new(parameters.harmonics_window_len());

        let candidate_generator = CandidateGenerator::new(
            parameters.candidate_generator_fft_len,
            parameters.half_interpolation_window_len,
            parameters.interpolation_factor,
            parameters.sample_rate,
            parameters.pitch_range.clone(),
        )?;
        let candidate_selector = CandidateSelector::new(
            parameters.candidate_selection_window_len(),
            parameters.candidate_taper,
            candidate_generator.normalized_candidate_frequencies(parameters.sample_rate, parameters.pitch_range.clone()),
        );
        Ok(Self {
            parameters,
            estimator,
            candidate_generator,
            candidate_selector,
        })
    }

    /// Process input from a queue of samples in a [`VecDeque`].
    ///
    /// As many samples as necessary to calculate the next pitch estimate are read from the [`VecDeque`], otherwise
    /// [`None`] is returned if more are required. To process as many samples as possible from the [`VecDeque`],
    /// `process` should be called repeatedly until [`None`] is returned.
    ///
    /// Input samples consumed are eventually removed from the front of the [`VecDeque`] by `process`, but a fixed-size
    /// window of past samples are left remaining in the [`VecDeque`] for access by later calls to `process` and are
    /// only removed when they are no longer needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use irapt::{Irapt, Parameters};
    /// use std::collections::VecDeque;
    /// use std::f64::consts::PI;
    ///
    /// let parameters = Parameters::default();
    /// let mut irapt = Irapt::new(parameters.clone()).expect("the default parameters should be valid");
    ///
    /// // Use a 100Hz sine wave as an example input signal
    /// let mut samples = (0..).map(|sample_index| f64::sin(sample_index as f64 / parameters.sample_rate * 2.0 * PI * 100.0));
    ///
    /// // Collect half of a second of input
    /// let mut sample_buffer = VecDeque::new();
    /// sample_buffer.extend(samples.by_ref().take(parameters.sample_rate as usize / 2));
    ///
    /// // Process as many samples as possible
    /// while let Some(estimated_pitch) = irapt.process(&mut sample_buffer) {
    ///     println!("estimated pitch: {}Hz with energy {}", estimated_pitch.frequency, estimated_pitch.energy);
    /// }
    ///
    /// // Simulate that half of a second more samples have become availoble and process them
    /// sample_buffer.extend(samples.by_ref().take(parameters.sample_rate as usize / 2));
    ///
    /// while let Some(estimated_pitch) = irapt.process(&mut sample_buffer) {
    ///     println!("estimated pitch: {}Hz with energy {}", estimated_pitch.frequency, estimated_pitch.energy);
    /// }
    /// ```
    pub fn process<S: Into<f64> + Copy>(&mut self, samples: &mut VecDeque<S>) -> Option<EstimatedPitch> {
        let initial_samples_len = samples.len();
        let step_len = self.parameters.step_len();
        while let (step_samples_len, Some(harmonics)) = (
            self.estimator.next_step_samples_len(),
            self.estimator.process_step(samples, step_len, self.parameters.sample_rate),
        ) {
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
                    .candidate_frequencies(self.parameters.sample_rate)
                    .nth(best_candidate_index)
                    .unwrap_or_else(|| panic!("candidate index out of bounds"));
                let removed_samples_count = initial_samples_len - samples.len();
                return Some(EstimatedPitch {
                    frequency: candidate_frequency,
                    energy:    (-min_candidate / energy.max(1e-4)),
                    index:     step_samples_len + removed_samples_count,
                });
            }
        }
        None
    }

    /// Resets all internal state associated with the sliding analysis window.
    ///
    /// The internal state after a reset is equivalent to that of a newly constructed [`Irapt`]. Resetting can be useful
    /// to avoid causing artifacts in the analysis when skipping a number of samples in the input without processing
    /// them.
    ///
    /// # Examples
    ///
    /// ```
    /// use irapt::{Irapt, Parameters};
    /// use std::collections::VecDeque;
    /// use std::f64::consts::PI;
    ///
    /// let parameters = Parameters::default();
    /// let mut irapt = Irapt::new(parameters.clone()).expect("the default parameters should be valid");
    ///
    /// // Use a 100Hz sine wave as an example input signal
    /// let mut samples = (0..).map(|sample_index| f64::sin(sample_index as f64 / parameters.sample_rate * 2.0 * PI * 100.0));
    ///
    /// // Collect half of a second of input
    /// let mut sample_buffer = VecDeque::new();
    /// sample_buffer.extend(samples.by_ref().take(parameters.sample_rate as usize / 2));
    ///
    /// while let Some(estimated_pitch) = irapt.process(&mut sample_buffer) {
    ///     println!("estimated pitch: {}Hz with energy {}", estimated_pitch.frequency, estimated_pitch.energy);
    /// }
    ///
    /// // Simulate that many more samples have become available
    /// let more_samples = samples.by_ref().take(parameters.sample_rate as usize * 10);
    ///
    /// // Reset irapt, clear the input buffer, skip all but half a second of input samples, and process the rest
    /// irapt.reset();
    /// sample_buffer.clear();
    /// sample_buffer.extend(more_samples.skip(parameters.sample_rate as usize * 19 / 2));
    ///
    /// while let Some(estimated_pitch) = irapt.process(&mut sample_buffer) {
    ///     println!("estimated pitch: {}Hz with energy {}", estimated_pitch.frequency, estimated_pitch.energy);
    /// }
    /// ```
    pub fn reset(&mut self) {
        self.estimator.reset();
    }
}

//
// Parameters impls
//

impl Parameters {
    /// Suggested default parameters.
    pub const DEFAULT: Self = Self {
        sample_rate: 6000.0,

        harmonics_estimation_interval:        0.005,
        harmonics_estimation_window_duration: 0.05,
        candidate_selection_window_duration:  0.3,

        pitch_range: 50.0..=450.0,

        candidate_generator_fft_len:   16384,
        half_interpolation_window_len: 12,
        interpolation_factor:          2,

        candidate_taper:      0.25,
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
