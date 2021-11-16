use alloc::boxed::Box;
use alloc::collections::{vec_deque, VecDeque};
use alloc::vec;
use core::mem;
use core::num::{NonZeroU16, NonZeroUsize};

use crate::buffer::InputBufferCursor;
use crate::fir_filter::hanning;
use crate::util::iter::IteratorExt;
use itertools::{chain, zip};
use num::Zero as _;

pub struct CandidateSelector {
    steps:                        VecDeque<Step>,
    last_step_voiced_min_costs:   Box<[f64]>,
    last_step_unvoiced_min_cost:  f64,
    last_step_itakura_distortion: f64,
    last_step_root_mean_square:   f64,
    voiced_min_costs_buffer:      Box<[f64]>,
    weights:                      Box<[f64]>,
}

#[derive(Clone, Copy, Debug)]
pub struct CandidateSelectionParameters {
    pub max_pitch_jump: usize,
    pub decay:          f64,
    pub voicing_state:  VoicingStateParameters,
}

#[derive(Clone, Copy, Debug)]
pub struct VoicingStateParameters {
    pub voiced_bias:                     f64,
    pub fixed_transition_cost:           f64,
    pub delta_amplitude_transition_cost: f64,
    pub delta_spectrum_transition_cost:  f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CandidateSelection {
    Unvoiced {
        energy: f64,
    },
    Voiced {
        selected_candidate_index: usize,
        energy:                   f64,
    }
}

#[derive(Clone)]
pub struct CandidateSelectionStepIter<'a> {
    max_pitch_jump: usize,
    steps:          vec_deque::Iter<'a, Step>,
    min_index:      CandidateSelection,
}

struct VoicingStateTransitionCosts {
    to_voiced:   f64,
    to_unvoiced: f64,
}

struct Step {
    voiced_min_indices: Box<[StepRelativeCandidateIndex]>,
    unvoiced_min_index: CandidateIndex,
    energy:             f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateIndex {
    Unvoiced,
    Voiced {
        one_based_index: NonZeroUsize,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StepRelativeCandidateIndex {
    Unvoiced,
    Voiced {
        one_based_index: NonZeroU16,
    },
}

impl CandidateSelector {
    pub fn new(
        steps_per_window: usize,
        taper: f64,
        normalized_candidate_frequencies: impl Iterator<Item = f64> + ExactSizeIterator,
    ) -> Self {
        let candidates_per_step = normalized_candidate_frequencies.len();
        let weights = normalized_candidate_frequencies
            .map(|normalized_frequency| 1.0 - (1.0 - normalized_frequency) * taper)
            .collect();
        Self {
            steps: VecDeque::with_capacity(steps_per_window),
            last_step_voiced_min_costs: vec![0.0; candidates_per_step].into(),
            last_step_unvoiced_min_cost: 0.0,
            last_step_itakura_distortion: 0.0,
            last_step_root_mean_square: 0.0,
            voiced_min_costs_buffer: vec![0.0; candidates_per_step].into(),
            weights,
        }
    }

    pub fn initialized(&self, steps_per_window: usize) -> bool {
        self.steps.len() == steps_per_window
    }

    pub fn process_step<S: Into<f64> + Copy>(
        &mut self,
        candidates: impl Iterator<Item = f64>,
        samples: &VecDeque<S>,
        cursor: &mut InputBufferCursor,
        step_len: usize,
        voicing_state_transition_window_len: usize,
        energy: f64,
        steps_per_window: usize,
        parameters: CandidateSelectionParameters,
    ) {
        let candidates_per_step = self.last_step_voiced_min_costs.len();

        let itakura_distortion = 1.0;

        let voicing_state_transition_window = {
            let available_window_len = voicing_state_transition_window_len.min(samples.len() - cursor.index());
            samples.range(cursor.index()..cursor.index() + available_window_len).map(|sample| (*sample).into())
        };
        let root_mean_square = root_mean_square(voicing_state_transition_window.clone(), voicing_state_transition_window_len);

        if let Some(remaining_len) = samples.len().checked_sub(voicing_state_transition_window.len()) {
            cursor.advance(step_len.min(remaining_len));
        }

        let root_mean_square_ratio = if root_mean_square.is_zero() && self.last_step_root_mean_square.is_zero() {
            1.0
        } else {
            root_mean_square / self.last_step_root_mean_square
        };

        let voicing_transition_costs = VoicingStateTransitionCosts::new(
            itakura_distortion,
            root_mean_square_ratio,
            &parameters.voicing_state,
        );

        let initialized = self.steps.len() == steps_per_window;
        let max_pitch_jump = parameters.max_pitch_jump;
        let decay = initialized.then(|| parameters.decay).unwrap_or(1.0);
        let recycled_step = initialized.then(|| self.steps.pop_front()).flatten();
        let mut new_step = recycled_step.unwrap_or_else(|| Step::new(candidates_per_step));

        // Calculate the globally minimal candidate from the current time step to calculate the total energy later.
        let mut min_candidate = f64::NAN;
        let candidates = candidates.inspect(|candidate| {
            min_candidate = candidate.min(min_candidate);
        });

        // Calculate the globally maximal candidate from the current time step, for adding to the unvoiced candidate cost below.
        let mut max_candidate: f64 = f64::NAN;
        let candidates = candidates.inspect(|candidate| {
            max_candidate = max_candidate.max(*candidate);
        });

        // Calculate locally minimal voiced candidate costs (within a window of size max_pitch_jump * 2) from the last time step.
        let unvoiced_to_voiced_cost = self.last_step_unvoiced_min_cost + voicing_transition_costs.to_voiced;
        let last_step_voiced_min_costs = to_voiced_min_candidates(
            &*self.last_step_voiced_min_costs,
            max_pitch_jump,
            unvoiced_to_voiced_cost,
        );

        // Apply pre-calculated weights to the candidates and add them to the best (minimal) candidates from the last time step.
        let candidate_costs = zip(candidates, &*self.weights).map(|(candidate, weight)| candidate * weight);

        let voiced_min_costs = zip(last_step_voiced_min_costs, candidate_costs)
            .map(|((min_index, min_cost), candidate_cost)| (min_index, (min_cost + candidate_cost) * decay));
        zip(&mut *new_step.voiced_min_indices, &mut *self.voiced_min_costs_buffer).set_from(voiced_min_costs);

        // Now that the maximum candidate from the current time step is calculated, calculate the unvoiced candidate cost for the current
        // time step.
        let unvoiced_candidate = -min_candidate + parameters.voicing_state.voiced_bias;
        let voiced_to_unvoiced_cost = unvoiced_candidate + voicing_transition_costs.to_unvoiced;
        let (unvoiced_min_index, last_step_unvoiced_min_cost) =
            to_unvoiced_min_candidate(&*self.last_step_voiced_min_costs, self.last_step_unvoiced_min_cost, voiced_to_unvoiced_cost);
        let unvoiced_min_cost = last_step_unvoiced_min_cost + unvoiced_candidate;
        new_step.unvoiced_min_index = unvoiced_min_index;

        // Save our calculations for later use by future time steps.
        self.last_step_unvoiced_min_cost = unvoiced_min_cost;
        self.last_step_itakura_distortion = itakura_distortion;
        self.last_step_root_mean_square = root_mean_square;
        mem::swap(&mut self.voiced_min_costs_buffer, &mut self.last_step_voiced_min_costs);
        self.steps.push_back(Step {
            voiced_min_indices: new_step.voiced_min_indices,
            unvoiced_min_index: new_step.unvoiced_min_index,
            energy: -min_candidate / energy.max(1e-4),
        });
    }

    pub fn best_candidate_steps(&self, steps_per_window: usize, max_pitch_jump: usize) -> Option<CandidateSelectionStepIter<'_>> {
        // follow trail of minimum indices through previous steps
        let initialized = self.steps.len() == steps_per_window;
        initialized.then(move || {
            let (voiced_min_index, voiced_min_cost) = (self.last_step_voiced_min_costs.iter())
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN encountered"))
                .unwrap_or_else(|| unreachable!());

            let energy = self.steps.back().map(|step| step.energy).unwrap_or_default();
            let min_index = if *voiced_min_cost < self.last_step_unvoiced_min_cost {
                CandidateSelection::Voiced { selected_candidate_index: voiced_min_index, energy }
            } else {
                CandidateSelection::Unvoiced { energy }
            };

            CandidateSelectionStepIter {
                max_pitch_jump,
                steps: self.steps.range(1..),
                min_index,
            }
        })
    }

    pub fn reset(&mut self) {
        self.steps.clear();
        self.last_step_voiced_min_costs.fill(0.0);
        self.last_step_unvoiced_min_cost = 0.0;
    }
}

impl Default for VoicingStateParameters {
    fn default() -> Self {
        Self {
            voiced_bias: 0.0,
            fixed_transition_cost: 0.005,
            delta_amplitude_transition_cost: 0.5,
            delta_spectrum_transition_cost: 0.5
        }
    }
}

impl VoicingStateTransitionCosts {
    fn new(itakura_distortion: f64, root_mean_square_ratio: f64, parameters: &VoicingStateParameters) -> Self {
        let shared = parameters.fixed_transition_cost +
            parameters.delta_spectrum_transition_cost * 0.2 / (itakura_distortion - 0.8);
        Self {
            to_unvoiced: shared + parameters.delta_amplitude_transition_cost * root_mean_square_ratio,
            to_voiced: match root_mean_square_ratio {
                ratio if ratio == f64::INFINITY => shared,
                _ => shared + parameters.delta_amplitude_transition_cost / root_mean_square_ratio,
            },
        }
    }
}

impl Iterator for CandidateSelectionStepIter<'_> {
    type Item = CandidateSelection;

    fn next(&mut self) -> Option<Self::Item> {
        let step = self.steps.next_back()?;

        self.min_index = match self.min_index {
            CandidateSelection::Voiced { selected_candidate_index: min_index, .. } => match step.voiced_min_indices[min_index] {
                StepRelativeCandidateIndex::Unvoiced => CandidateSelection::Unvoiced {
                    energy: step.energy,
                },
                StepRelativeCandidateIndex::Voiced { one_based_index: relative_one_based_index } =>
                    CandidateSelection::Voiced {
                        selected_candidate_index: (min_index.saturating_sub(self.max_pitch_jump) +
                                                   (relative_one_based_index.get() - 1) as usize),
                        energy: step.energy,
                    },
            },
            CandidateSelection::Unvoiced { .. } => CandidateSelection::Unvoiced {
                energy: step.energy,
            },
        };

        Some(self.min_index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.steps.size_hint()
    }
}

impl ExactSizeIterator for CandidateSelectionStepIter<'_> {}

impl Step {
    fn new(candidates_per_step: usize) -> Self {
        Self {
            voiced_min_indices: vec![StepRelativeCandidateIndex::Unvoiced; candidates_per_step].into(),
            unvoiced_min_index: CandidateIndex::Unvoiced,
            energy:             f64::NAN,
        }
    }
}

impl CandidateIndex {
    fn new_voiced(index: usize) -> Self {
        Self::Voiced {
            one_based_index: NonZeroUsize::new(index.saturating_add(1)).unwrap_or_else(|| unreachable!()),
        }
    }
}

impl StepRelativeCandidateIndex {
    fn new_voiced(relative_index: u16) -> Self {
        Self::Voiced {
            one_based_index: NonZeroU16::new(relative_index.saturating_add(1)).unwrap_or_else(|| unreachable!()),
        }
    }
}

fn to_unvoiced_min_candidate(
    last_step_voiced_candidates: &[f64],
    last_step_unvoiced_candidate: f64,
    voiced_to_unvoiced_cost: f64,
) -> (CandidateIndex, f64) {
    let (voiced_global_min_index, voiced_global_min_cost) = min_candidate_cost(last_step_voiced_candidates)
        .unwrap_or_else(|| unreachable!());
    let voiced_min_cost = voiced_global_min_cost + voiced_to_unvoiced_cost;
    if voiced_min_cost < last_step_unvoiced_candidate {
        (CandidateIndex::new_voiced(voiced_global_min_index), voiced_min_cost)
    } else {
        (CandidateIndex::Unvoiced, last_step_unvoiced_candidate)
    }
}

fn to_voiced_min_candidates(
    voiced_candidate_costs: &[f64],
    max_pitch_jump: usize,
    unvoiced_to_voiced_cost: f64,
) -> impl Iterator<Item = (StepRelativeCandidateIndex, f64)> + '_ {
    // Calculate locally minimal voiced candidates (within a window of size max_pitch_jump * 2) from the last time step.
    let voiced_local_min_candidates = windows_inexact(voiced_candidate_costs, max_pitch_jump + 1, max_pitch_jump * 2 + 1)
        .map(min_candidate_cost)
        .map(|min_candidate_cost| min_candidate_cost.unwrap_or_else(|| unreachable!()))
        .map(|(min_index, min_cost)| (StepRelativeCandidateIndex::new_voiced(min_index as u16), min_cost));

    // Decide whether the unvoiced candidate from the last time step is better (i.e. less cost) than the locally minimal voiced
    // candidates from the last time step.
    voiced_local_min_candidates.map(move |(min_index, min_cost)| {
        if min_cost < unvoiced_to_voiced_cost {
            (min_index, min_cost)
        } else {
            (StepRelativeCandidateIndex::Unvoiced, unvoiced_to_voiced_cost)
        }
    })
}

fn min_candidate_cost<'a, I: IntoIterator<Item = &'a f64>>(candidates: I) -> Option<(usize, f64)> {
    candidates
        .into_iter()
        .cloned()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN encountered"))
}

fn windows_inexact<T>(slice: &[T], min_window_len: usize, max_window_len: usize) -> impl Iterator<Item = &[T]> {
    let beginning = (min_window_len..max_window_len).map(move |len| &slice[..len]);
    let middle = slice.windows(max_window_len);
    let end = (min_window_len..max_window_len).rev().map(move |len| &slice[slice.len() - len..]);
    chain!(beginning, middle, end)
}

fn root_mean_square(samples: impl Iterator<Item = f64> + ExactSizeIterator, window_len: usize) -> f64 {
    let hanning = (0..).map(|index| hanning(index, window_len as u32));
    let windowed = zip(samples, hanning).map(|(sample, window)| sample * window);
    let squared = windowed.map(|value| value * value);
    f64::sqrt(squared.sum::<f64>() / window_len as f64)
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;
    use itertools::repeat_n;

    use super::super::generation;
    use super::*;

    #[test]
    fn test_process_step() {
        let steps_per_window = 5;
        let energy = 0.0;
        let parameters = CandidateSelectionParameters {
            max_pitch_jump: 23,
            decay:          0.95,
            voicing_state:  VoicingStateParameters::default(),
        };

        let generator = generation::test::test_process_step_candidate_generator();
        let normalized_candidate_frequencies =
            generator.normalized_candidate_frequencies(generation::test::TEST_SAMPLE_RATE, generation::test::TEST_PITCH_RANGE);
        let steps_candidates = generation::test::test_process_step_expected();

        let expected = [96, 96, 96, 94, 94, 97, 94, 94];
        let expected = repeat_n(None, steps_per_window - 1)
            .chain(expected.iter().cloned().map(|index| Some(Some(index))))
            .collect::<Vec<_>>();

        let mut selector = CandidateSelector::new(steps_per_window, 0.25, normalized_candidate_frequencies);
        let best_candidate_indices = steps_candidates
            .map(|step_candidates| {
                selector.process_step(
                    step_candidates,
                    &VecDeque::<f64>::default(),
                    &mut InputBufferCursor::default(),
                    1,
                    1,
                    energy,
                    steps_per_window,
                    parameters
                );
                selector.best_candidate_steps(steps_per_window, parameters.max_pitch_jump)
                        .and_then(|best_candidate_steps| best_candidate_steps.last())
                        .map(|step_best_candidate| match step_best_candidate {
                            CandidateSelection::Voiced { selected_candidate_index, .. } => Some(selected_candidate_index),
                            CandidateSelection::Unvoiced { .. } => None,
                        })
            })
            .collect::<Vec<_>>();
        assert_eq!(best_candidate_indices, expected);
    }
}
