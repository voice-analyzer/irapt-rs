use core::f64::consts::PI;

/// Computes the value at given `index` of a Hamming window of length `len`.
pub fn hamming(index: u32, len: u32) -> f64 {
    0.54 - 0.46 * f64::cos(2.0 * PI * f64::from(index) / f64::from(len - 1))
}

/// Computes the value at given `index` of a Hanning window of length `len`.
pub fn hanning(index: u32, len: u32) -> f64 {
    0.5 - 0.5 * f64::cos(2.0 * PI * f64::from(index) / f64::from(len - 1))
}

/// Computes the values of a hamming-window FIR lowpass filter of length `len` with given `cutoff_frequency`.
pub fn lowpass_fir_filter(len: u32, cutoff_frequency: f64, window: fn(u32, u32) -> f64) -> impl Iterator<Item = f64> + ExactSizeIterator {
    let center = f64::from(len - 1) / 2.0;
    (0..len).map(move |index| lowpass_fir_rect(index, center, cutoff_frequency) * window(index, len))
}

/// Scales a lowpass FIR filter so the center of the pass band has magnitude of exactly one.
pub fn scale_lowpass_filter(filter: &mut [f64]) {
    let sum = filter.iter().sum::<f64>();
    filter.iter_mut().for_each(|filter_value| *filter_value /= sum);
}

/// Computes the value at given `index` of a rectangular-window FIR lowpass filter centered at index `center` with given
/// `cutoff_frequency`.
pub fn lowpass_fir_rect(index: u32, center: f64, cutoff_frequency: f64) -> f64 {
    let k = f64::abs(f64::from(index) - center);
    cutoff_frequency * sinc(k * cutoff_frequency)
}

/// Computes the sinc of a number (in radians).
pub fn sinc(value: f64) -> f64 {
    if value == 0.0 { 1.0 } else { f64::sin(value * PI) / (value * PI) }
}

#[cfg(test)]
mod test;
