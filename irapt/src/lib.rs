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
//! while let (initial_sample_buffer_len, Some(output)) = (
//!     sample_buffer.len(),
//!     irapt.process(&mut sample_buffer),
//! ) {
//!     let estimated_pitch = output.pitch_estimates().final_estimate();
//!     let estimated_pitch_index = (sample_index as isize + estimated_pitch.offset) as usize;
//!     let estimated_pitch_time = estimated_pitch_index as f64 / parameters.sample_rate;
//!     if let Some(estimated_pitch_frequency) = estimated_pitch.frequency {
//!         println!("estimated pitch at {:0.3}: {}Hz with energy {}",
//!                  estimated_pitch_time, estimated_pitch_frequency, estimated_pitch.energy);
//!     }
//!     sample_index += initial_sample_buffer_len - sample_buffer.len();
//! }
//! ```

extern crate alloc;

#[macro_use]
mod util;

#[doc(hidden)]
pub mod buffer;
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

use self::buffer::{InputBufferCursor, InputBufferCursors};
use self::candidates::{
    CandidateFrequencyIter, CandidateGenerator, CandidateSelection, CandidateSelectionStepIter, CandidateSelector,
    CandidateSelectionParameters, VoicingStateParameters,
};
use self::error::InvalidParameterError;
use self::harmonics::HarmonicParametersEstimator;

use alloc::collections::VecDeque;
use core::array;
use core::iter::Enumerate;
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
    cursors:             InputBufferCursors<IraptInputBufferCursors>,
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

    /// Amount of bias toward deciding that voiced speech is present as opposed to unvoiced spech or background noise.
    ///
    /// The suggested default is `0.0`.
    pub voiced_bias: f64,
}

/// The output of [`Irapt::process`].
///
/// This `struct` holds the output, including estimated pitches, of the IRAPT algorithm after processing a single time step, and is created
/// by the [`process`](Irapt::process) method on [`Irapt`]. See its documentation for more.
pub struct Output<'a> {
    estimated_pitches: EstimatedPitchIter<'a>,
    more_output: bool,
}

/// An estimate of the pitch in the input at a specific sample offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimatedPitch {
    /// Frequency, in Hz, of the estimated pitch.
    pub frequency: Option<f64>,
    /// Arbitrary measure, from `0.0..`, of the energy associated with the estimated pitch.
    pub energy:    f64,
    /// The offset in samples within the input buffer (_before_ removal of consumed samples) at which the pitch was estimated. This may be
    /// negative, since estimates can be returned for samples which have already been removed from the input buffer.
    pub offset:    isize,
}

/// An iterator over pitches estimated over time in the input, in reverse chronological order.
///
/// This `struct` is created by the [`pitch_estimates`](Output::pitch_estimates) method on [`Output`]. See its documentation for more.
#[derive(Clone)]
pub struct EstimatedPitchIter<'a> {
    selected_candidates: Enumerate<CandidateSelectionStepIter<'a>>,
    candidate_frequencies: CandidateFrequencyIter,
    last_step_sample_index: usize,
    step_len: usize,
}

#[derive(Default)]
struct IraptInputBufferCursors {
    harmonics: InputBufferCursor,
    candidates: InputBufferCursor,
}

const VOICING_STATE_TRANSITION_WINDOW_DURATION: f64 = 0.3;

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
            cursors: Default::default(),
            estimator,
            candidate_generator,
            candidate_selector,
        })
    }

    /// Returns the `Parameters` specified during construction.
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
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
    /// while let Some(output) = irapt.process(&mut sample_buffer) {
    ///     let estimated_pitch = output.pitch_estimates().final_estimate();
    ///     if let Some(estimated_pitch_frequency) = estimated_pitch.frequency {
    ///         println!("estimated pitch: {}Hz with energy {}", estimated_pitch_frequency, estimated_pitch.energy);
    ///     }
    /// }
    ///
    /// // Simulate that half of a second more samples have become availoble and process them
    /// sample_buffer.extend(samples.by_ref().take(parameters.sample_rate as usize / 2));
    ///
    /// while let Some(output) = irapt.process(&mut sample_buffer) {
    ///     let estimated_pitch = output.pitch_estimates().final_estimate();
    ///     if let Some(estimated_pitch_frequency) = estimated_pitch.frequency {
    ///         println!("estimated pitch: {}Hz with energy {}", estimated_pitch_frequency, estimated_pitch.energy);
    ///     }
    /// }
    /// ```
    pub fn process<S: Into<f64> + Copy>(&mut self, samples: &mut VecDeque<S>) -> Option<Output<'_>> {
        let step_len = self.parameters.step_len();
        let voicing_state_transition_window_len = self.parameters.voicing_state_transition_window_len();

        let cursors = self.cursors.cursors_mut();

        let mut processed_step_sample_index = None;
        while let (step_sample_index, Some(harmonics)) = (
            self.estimator.next_step_samples_len(),
            self.estimator.process_step(samples, &mut cursors.harmonics, step_len, self.parameters.sample_rate),
        ) {
            let mut energy = 0.0;
            let harmonics = harmonics.inspect(|harmonic| {
                energy += harmonic.amplitude * harmonic.amplitude;
            });

            self.candidate_generator.process_step_harmonics(harmonics, self.parameters.sample_rate);
            let candidates = self.candidate_generator.generate_step_candidates();

            let candidate_selection_parameters = CandidateSelectionParameters {
                max_pitch_jump: self.parameters.candidate_max_jump,
                decay:          self.parameters.candidate_step_decay,
                voicing_state:  VoicingStateParameters {
                    voiced_bias: self.parameters.voiced_bias,
                    ..Default::default()
                },
            };

            self.candidate_selector.process_step(
                candidates,
                &samples,
                &mut cursors.candidates,
                step_len,
                voicing_state_transition_window_len,
                energy,
                self.parameters.candidate_selection_window_len(),
                candidate_selection_parameters,
            );

            if self.candidate_selector.initialized(self.parameters.candidate_selection_window_len()) {
                processed_step_sample_index = Some(step_sample_index);
                break;
            }
        }

        self.cursors.advance_buffer(samples);

        let last_step_sample_index = processed_step_sample_index?;
        let more_output = samples.len() >= self.cursors.cursors_mut().harmonics.index() + self.estimator.next_step_samples_len();

        let selected_candidates = self.candidate_selector.best_candidate_steps(
            self.parameters.candidate_selection_window_len(),
            self.parameters.candidate_max_jump,
        );
        Some(Output {
            estimated_pitches: EstimatedPitchIter {
                selected_candidates: selected_candidates?.enumerate(),
                candidate_frequencies: self.candidate_generator.candidate_frequencies(self.parameters.sample_rate),
                last_step_sample_index,
                step_len: self.parameters.step_len(),
            },
            more_output,
        })
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
    /// while let Some(output) = irapt.process(&mut sample_buffer) {
    ///     let estimated_pitch = output.pitch_estimates().final_estimate();
    ///     if let Some(estimated_pitch_frequency) = estimated_pitch.frequency {
    ///         println!("estimated pitch: {}Hz with energy {}", estimated_pitch_frequency, estimated_pitch.energy);
    ///     }
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
    /// while let Some(output) = irapt.process(&mut sample_buffer) {
    ///     let estimated_pitch = output.pitch_estimates().final_estimate();
    ///     if let Some(estimated_pitch_frequency) = estimated_pitch.frequency {
    ///         println!("estimated pitch: {}Hz with energy {}", estimated_pitch_frequency, estimated_pitch.energy);
    ///     }
    /// }
    /// ```
    pub fn reset(&mut self) {
        self.estimator.reset();
        self.candidate_selector.reset();
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
        voiced_bias:          0.0,
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

    fn voicing_state_transition_window_len(&self) -> usize {
        (VOICING_STATE_TRANSITION_WINDOW_DURATION * self.sample_rate).round() as usize
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self::DEFAULT
    }
}

//
// Output impls
//

impl Output<'_> {
    /// Returns whether further output can be produced given the input samples.
    ///
    /// More output can be produced by calling [`Irapt::process`].
    pub fn more_output(&self) -> bool {
        self.more_output
    }

    /// Returns all pitch estimates for the given input, including both those tentative and final, in reverse chronological order.
    ///
    /// All but the last of the yielded pitches are tentative estimates calculated up to [`candidate_selection_window_duration`] seconds in
    /// the past. The estimates are returned in reverse chronological order. The exact sample offsets for the estimates are returned in
    /// [`EstimatedPitch::offset`].
    ///
    /// The last estimate yielded is final for the given time offset. It can also be retrieved by calling [`final_estimate`] on the returned
    /// iterator.
    ///
    /// [`candidate_selection_window_duration`]: Parameters::candidate_selection_window_duration
    /// [`final_estimate`]: EstimatedPitchIter::final_estimate
    pub fn pitch_estimates(&self) -> EstimatedPitchIter<'_> {
        self.estimated_pitches.clone()
    }
}

//
// EstimatedPitchIter impls
//

impl EstimatedPitchIter<'_> {
    /// Returns a final pitch estimate for the given input, at a time delay.
    ///
    /// The returned pitch is the final estimate calculated at approximately [`candidate_selection_window_duration`] seconds in the past.
    /// The exact sample offset for the estimate is returned in [`EstimatedPitch::offset`].
    ///
    /// [`candidate_selection_window_duration`]: Parameters::candidate_selection_window_duration
    pub fn final_estimate(self) -> EstimatedPitch {
        self.last().unwrap_or_else(|| unreachable!())
    }
}

impl Iterator for EstimatedPitchIter<'_> {
    type Item = EstimatedPitch;

    fn next(&mut self) -> Option<Self::Item> {
        let (step_index, candidate_selection) = self.selected_candidates.next()?;
        let offset = self.last_step_sample_index.wrapping_sub((1 + step_index) * self.step_len) as isize;
        let pitch = match candidate_selection {
            CandidateSelection::Unvoiced { energy } => EstimatedPitch {
                frequency: None,
                energy,
                offset,
            },
            CandidateSelection::Voiced { selected_candidate_index, energy } => {
                let frequency = (self.candidate_frequencies.clone())
                    .nth(selected_candidate_index)
                    .unwrap_or_else(|| panic!("candidate index out of bounds"));
                EstimatedPitch {
                    frequency: Some(frequency),
                    energy,
                    offset,
                }
            }
        };
        Some(pitch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.selected_candidates.size_hint()
    }
}

impl ExactSizeIterator for EstimatedPitchIter<'_> {}

//
// Irapt impls
//

impl<'a> IntoIterator for &'a mut IraptInputBufferCursors {
    type Item = &'a mut InputBufferCursor;
    type IntoIter = array::IntoIter<&'a mut InputBufferCursor, 2>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new([&mut self.harmonics, &mut self.candidates])
    }

}
