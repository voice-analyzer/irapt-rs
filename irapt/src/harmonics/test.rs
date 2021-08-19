use core::f64::consts::PI;

use itertools::zip_eq;

use crate::util::test::parse_csv;

use super::*;

pub fn test_process_step_expected() -> impl Iterator<Item = impl Iterator<Item = HarmonicParameter>> {
    let expected_step_amplitudes = parse_csv(include_bytes!("test/process_step/amplitudes.csv"));
    let expected_step_frequencies = parse_csv(include_bytes!("test/process_step/frequencies.csv"));
    let expected_steps = zip_eq(expected_step_amplitudes, expected_step_frequencies).map(|(amplitudes, frequencies)| {
        zip_eq(amplitudes, frequencies).map(|(amplitude, frequency)| HarmonicParameter { amplitude, frequency })
    });
    expected_steps
}

#[test]
fn test_process_step() {
    let expected_steps = test_process_step_expected();

    let window_len = 316;
    let step_len = 32;
    let sample_rate = 6000.0;
    let signal_frequency = 100.0;

    let mut estimator = HarmonicParametersEstimator::new(window_len);

    let mut samples = (0..).map(|sample_index| f64::sin(sample_index as f64 / sample_rate * 2.0 * PI * signal_frequency) as f32);
    let mut sample_buffer = samples.by_ref().take(estimator.next_step - step_len).collect::<VecDeque<_>>();

    for (step_index, expected_step) in expected_steps.enumerate() {
        sample_buffer.extend(samples.by_ref().take(step_len));
        let harmonics = estimator.process_step(&mut sample_buffer, step_len, sample_rate).unwrap();
        assert_iter_approx_eq!(harmonics, expected_step, 1e-7%, .amplitude .frequency, "step {}", step_index);
    }
    assert!(estimator.process_step(&mut sample_buffer, step_len, sample_rate).is_none());
}
