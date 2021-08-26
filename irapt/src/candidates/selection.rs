use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec;
use core::mem;

use itertools::{chain, izip};

pub struct CandidateSelector {
    steps:                VecDeque<Step>,
    last_step_min_values: Box<[f64]>,
    min_values_buffer:    Box<[f64]>,
    weights:              Box<[f64]>,
}

struct Step {
    min_indices: Box<[u16]>,
}

impl CandidateSelector {
    pub fn new(steps_per_window: usize, taper: f64, normalized_candidate_frequencies: impl Iterator<Item = f64> + ExactSizeIterator) -> Self {
        let candidates_per_step = normalized_candidate_frequencies.len();
        let weights = normalized_candidate_frequencies
            .map(|normalized_frequency| 1.0 - (1.0 - normalized_frequency) * taper)
            .collect();
        Self {
            steps:                VecDeque::with_capacity(steps_per_window),
            last_step_min_values: vec![0.0; candidates_per_step].into(),
            min_values_buffer:    vec![0.0; candidates_per_step].into(),
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
        let candidates_per_step = self.last_step_min_values.len();

        let initialized = self.steps.len() == steps_per_window;
        let decay = initialized.then(|| decay).unwrap_or(1.0);
        let recycled_step = initialized.then(|| self.steps.pop_front()).flatten();
        let mut new_step = recycled_step.unwrap_or_else(|| Step::new(candidates_per_step));

        let last_step_min_values_windows = windows_inexact(&*self.last_step_min_values, max_pitch_jump + 1, max_pitch_jump * 2 + 1);
        let new_min_values = &mut *self.min_values_buffer;
        let new_min_indices = &mut *new_step.min_indices;
        for (last_step_min_values_window, candidate, weight, new_min_value, new_min_index) in
            izip!(last_step_min_values_windows, candidates, &*self.weights, new_min_values, new_min_indices)
        {
            let (min_index, min_value) = min_candidate(last_step_min_values_window).unwrap_or_else(|| unreachable!());
            *new_min_value = (candidate * weight + min_value) * decay;
            *new_min_index = min_index as u16;
        }

        mem::swap(&mut self.min_values_buffer, &mut self.last_step_min_values);
        self.steps.push_back(new_step);
        let initialized = self.steps.len() == steps_per_window;

        // follow trail of minimum indices through previous steps
        initialized.then(|| {
            let (min_index, _) = min_candidate(&*self.last_step_min_values).unwrap_or_else(|| unreachable!());
            self.steps.range(1..).rev().fold(min_index, |min_index, step| {
                min_index.saturating_sub(max_pitch_jump) + step.min_indices[min_index] as usize
            })
        })
    }

    pub fn reset(&mut self) {
        self.steps.clear();
        self.last_step_min_values.fill(0.0);
    }
}

impl Step {
    fn new(candidates_per_step: usize) -> Self {
        Self {
            min_indices: vec![0; candidates_per_step].into(),
        }
    }
}

fn min_candidate<'a, I: IntoIterator<Item = &'a f64>>(candidates: I) -> Option<(usize, f64)> {
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
