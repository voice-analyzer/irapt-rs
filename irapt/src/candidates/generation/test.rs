use super::*;
use crate::harmonics;
use crate::util::test::parse_csv;

use alloc::vec::Vec;
use matches::assert_matches;

pub fn test_process_step_expected() -> impl Iterator<Item = impl Iterator<Item = f64> + ExactSizeIterator> {
    parse_csv(include_bytes!("test/process_step/candidates.csv"))
}

#[test]
fn test_process_step() {
    let expected_steps = test_process_step_expected();

    let sample_rate = 6000.0;

    let mut harmonics = harmonics::test::test_process_step_expected();
    let mut candidate_generator = CandidateGenerator::new(16384, 12, 2, sample_rate, 50.0..=450.0).unwrap();

    for (step_index, expected_step) in expected_steps.enumerate() {
        let candidates = candidate_generator.process_step(harmonics.next().unwrap(), sample_rate);
        let candidates = candidates.collect::<Vec<_>>();
        assert_iter_approx_eq!(candidates, expected_step, 1e-7%, "step {}", step_index);
    }
}

#[test]
fn test_interpolation_window_too_long() {
    assert!(CandidateGenerator::new(16384, 12, 2, 6000.0, 6000.0 / (16384.0 - 12.0)..=6000.0 / 12.0).is_ok());

    assert_matches!(
        CandidateGenerator::new(16384, 12, 2, 6000.0, 6000.0 / (16384.0 - 12.0)..=6000.0 / 11.0).err().unwrap(),
        InvalidParameterError::InterpolationWindowTooLong { .. }
    );
    assert_matches!(
        CandidateGenerator::new(16384, 12, 2, 6000.0, 6000.0 / (16384.0 - 11.0)..=6000.0 / 12.0).err().unwrap(),
        InvalidParameterError::InterpolationWindowTooLong { .. }
    );
}

#[test]
fn test_candidate_frequency() {
    let mut expected = parse_csv::<f64>(include_bytes!("test/candidate_frequency/frequencies.csv"));
    let expected = expected.next().unwrap();

    let sample_rate = 6000.0;

    let candidate_generator = CandidateGenerator::new(16384, 12, 2, sample_rate, 50.0..=450.0).unwrap();

    let candidate_frequencies = (0..candidate_generator.window_len())
        .map(|candidate_index| candidate_generator.candidate_frequency(candidate_index, sample_rate));

    assert_iter_approx_eq!(candidate_frequencies, expected, 1e-12);
}
