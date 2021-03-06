//! Error types returned by `irapt`.

/// An error returned when the supplied [`Parameters`](crate::Parameters) are invalid or conflict with each other.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum InvalidParameterError {
    /// The supplied `half_interploation_window_len` is too large for the given `pitch_range`, `sample_rate`, and
    /// `fft_len`.
    InterpolationWindowTooLong {
        /// The maximum allowable length of `half_interpolation_window_len` for the given `pitch_range` and
        /// `sample_rate`.
        max_length: u32,
    },
}
