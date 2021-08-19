use core::f32::consts::PI;

use crate::util::test::parse_csv;

use super::*;

#[test]
fn test_process() {
    let mut expected = parse_csv::<Complex<f64>>(include_bytes!("test/process/filtered.csv"));
    let expected = expected.next().unwrap();

    let window_len = 316;
    let channel_count = window_len + 44;
    let signal_frequency = 10.0;
    let mut channels = vec![<_>::default(); channel_count as usize];
    let mut filter = PolyphaseFilter::new(window_len, channels.len());
    let samples = (0..channel_count)
        .map(|sample_index| f32::sin(sample_index as f32 / channel_count as f32 * 2.0 * PI * signal_frequency))
        .rev();
    filter.process(samples.map(f64::from), &mut channels);
    assert_iter_approx_eq!(channels, expected, 1e-7, .re .im);
}
