use crate::util::test::parse_csv;

use super::*;

#[test]
fn test_lowpass_fir_filter() {
    // expected values obtained from matlab with `fir1(316, 1/360, hamming(317), 'noscale')`
    let mut expected = parse_csv::<f64>(include_bytes!("test/lowpass_fir_filter/expected.csv"));
    let expected = expected.next().unwrap();
    let actual = lowpass_fir_filter(expected.len() as u32, 1.0 / 360.0, hamming);
    assert_iter_approx_eq!(actual, expected, 1e-15);
}

#[test]
fn test_lowpass_fir_rect() {
    // expected values obtained from matlab with `firls(48, [0; 0.5; 0.5; 1], [1; 1; 0; 0])`
    let mut expected = parse_csv::<f64>(include_bytes!("test/lowpass_fir_rect/expected.csv"));
    let expected = expected.next().unwrap();
    let actual = (0..49).map(|index| lowpass_fir_rect(index, 24.0, 0.5));
    assert_iter_approx_eq!(actual, expected, 1e-15);
}

#[test]
fn test_hamming() {
    // expected values obtained from matlab with `hamming(49)`
    let expected = parse_csv::<f64>(include_bytes!("test/hamming/expected.csv")).next().unwrap();
    let actual = (0..49).map(|index| hamming(index, 49));
    assert_iter_approx_eq!(actual, expected, 1e-15);
}
