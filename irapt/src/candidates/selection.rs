use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec;
use core::mem;

use crate::util::iter::IteratorExt;
use itertools::{chain, zip};

pub struct CandidateSelector {
    steps:               VecDeque<Step>,
    last_step_min_costs: Box<[f64]>,
    min_costs_buffer:    Box<[f64]>,
    weights:             Box<[f64]>,
}

struct Step {
    min_indices: Box<[u16]>,
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
            last_step_min_costs: vec![0.0; candidates_per_step].into(),
            min_costs_buffer: vec![0.0; candidates_per_step].into(),
            weights,
        }
    }

    pub fn process_step(
        &mut self,
        candidates: impl Iterator<Item = f64>,
        steps_per_window: usize,
        max_pitch_jump: usize,
        decay: f64,
    ) -> Option<usize> {
        let candidates_per_step = self.last_step_min_costs.len();

        let initialized = self.steps.len() == steps_per_window;
        let decay = initialized.then(|| decay).unwrap_or(1.0);
        let recycled_step = initialized.then(|| self.steps.pop_front()).flatten();
        let mut new_step = recycled_step.unwrap_or_else(|| Step::new(candidates_per_step));

        // Calculate locally minimal candidate costs (within a window of size max_pitch_jump * 2) from the last time step.
        let last_step_min_costs = min_candidate_costs(&*self.last_step_min_costs, max_pitch_jump);

        // Apply pre-calculated weights to the candidates and add them to the best (minimal) candidates from the last time step.
        let weighted_candidates = zip(candidates, &*self.weights).map(|(candidate, weight)| candidate * weight);
        let min_costs = zip(last_step_min_costs, weighted_candidates)
            .map(|((min_index, min_cost), weighted_candidate)| (min_index, (min_cost + weighted_candidate) * decay));

        // Save our calculations for later use by future time steps.
        zip(&mut *new_step.min_indices, &mut *self.min_costs_buffer).set_from(min_costs);
        mem::swap(&mut self.min_costs_buffer, &mut self.last_step_min_costs);
        self.steps.push_back(new_step);
        let initialized = self.steps.len() == steps_per_window;

        // follow trail of minimum indices through previous steps
        initialized.then(|| {
            let (min_index, _) = min_candidate_cost(&*self.last_step_min_costs).unwrap_or_else(|| unreachable!());
            self.steps.range(1..).rev().fold(min_index, |min_index, step| {
                min_index.saturating_sub(max_pitch_jump) + step.min_indices[min_index] as usize
            })
        })
    }

    pub fn reset(&mut self) {
        self.steps.clear();
        self.last_step_min_costs.fill(0.0);
    }
}

impl Step {
    fn new(candidates_per_step: usize) -> Self {
        Self {
            min_indices: vec![0; candidates_per_step].into(),
        }
    }
}

fn min_candidate_costs(candidate_costs: &[f64], max_pitch_jump: usize) -> impl Iterator<Item = (u16, f64)> + '_ {
    // Calculate locally minimal candidate costs (within a window of size max_pitch_jump * 2) from the last time step.
    windows_inexact(candidate_costs, max_pitch_jump + 1, max_pitch_jump * 2 + 1)
        .map(min_candidate_cost)
        .map(|min_candidate_cost| min_candidate_cost.unwrap_or_else(|| unreachable!()))
        .map(|(min_index, min_cost)| (min_index as u16, min_cost))
}

fn min_candidate_cost<'a, I: IntoIterator<Item = &'a f64>>(candidate_costs: I) -> Option<(usize, f64)> {
    candidate_costs
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

#[cfg(test)]
mod test {
    use alloc::vec::Vec;
    use itertools::repeat_n;

    use super::super::generation;
    use super::*;

    #[test]
    fn test_process_step() {
        let steps_per_window = 5;
        let max_pitch_jump = 23;
        let decay = 0.95;

        let generator = generation::test::test_process_step_candidate_generator();
        let normalized_candidate_frequencies =
            generator.normalized_candidate_frequencies(generation::test::TEST_SAMPLE_RATE, generation::test::TEST_PITCH_RANGE);
        let steps_candidates = generation::test::test_process_step_expected();

        let expected = [96, 96, 96, 94, 94, 97, 94, 94];
        let expected = repeat_n(None, steps_per_window - 1)
            .chain(expected.iter().cloned().map(Some))
            .collect::<Vec<_>>();

        let mut selector = CandidateSelector::new(steps_per_window, 0.25, normalized_candidate_frequencies);
        let best_candidate_indices = steps_candidates
            .map(|step_candidates| selector.process_step(step_candidates, steps_per_window, max_pitch_jump, decay))
            .collect::<Vec<_>>();
        assert_eq!(best_candidate_indices, expected);
    }
}
