use super::*;
use crate::harmonics;
use crate::util::test::parse_csv;

use alloc::vec::Vec;
use matches::assert_matches;

pub const TEST_SAMPLE_RATE: f64 = 6000.0;
pub const TEST_PITCH_RANGE: RangeInclusive<f64> = 50.0..=450.0;

pub fn test_process_step_candidate_generator() -> CandidateGenerator {
    CandidateGenerator::new(16384, 12, 2, TEST_SAMPLE_RATE, TEST_PITCH_RANGE).unwrap()
}

pub fn test_process_step_expected() -> impl Iterator<Item = impl Iterator<Item = f64> + ExactSizeIterator> {
    parse_csv(include_bytes!("test/process_step/candidates.csv"))
}

#[test]
fn test_process_step() {
    let expected_steps = test_process_step_expected();

    let mut harmonics = harmonics::test::test_process_step_expected();
    let mut candidate_generator = test_process_step_candidate_generator();

    for (step_index, expected_step) in expected_steps.enumerate() {
        let candidates = candidate_generator.process_step(harmonics.next().unwrap(), TEST_SAMPLE_RATE);
        let candidates = candidates.collect::<Vec<_>>();
        assert_iter_approx_eq!(candidates, expected_step, 1e-7%, "step {}", step_index);
    }
}

#[test]
fn test_interpolation_window_too_long() {
    assert!(CandidateGenerator::new(16384, 12, 2, 6000.0, 6000.0 / (16384.0 - 12.0)..=6000.0 / 12.0).is_ok());

    assert_matches!(
        CandidateGenerator::new(16384, 12, 2, 6000.0, 6000.0 / (16384.0 - 12.0)..=6000.0 / 11.0)
            .err()
            .unwrap(),
        InvalidParameterError::InterpolationWindowTooLong { .. }
    );
    assert_matches!(
        CandidateGenerator::new(16384, 12, 2, 6000.0, 6000.0 / (16384.0 - 11.0)..=6000.0 / 12.0)
            .err()
            .unwrap(),
        InvalidParameterError::InterpolationWindowTooLong { .. }
    );
}

#[test]
fn test_candidate_frequency() {
    let mut expected = parse_csv::<f64>(include_bytes!("test/candidate_frequency/frequencies.csv"));
    let expected = expected.next().unwrap();

    let sample_rate = 6000.0;

    let candidate_generator = CandidateGenerator::new(16384, 12, 2, sample_rate, 50.0..=450.0).unwrap();

    assert_iter_approx_eq!(candidate_generator.candidate_frequencies(sample_rate), expected, 1e-12);
}
