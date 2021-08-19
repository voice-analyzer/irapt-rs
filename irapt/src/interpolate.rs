use alloc::boxed::Box;
use alloc::collections::VecDeque;

use crate::fir_filter::{hamming, lowpass_fir_filter, scale_lowpass_filter};

use itertools::{zip, Itertools};

pub struct InterpolationFilter {
    factor: u8,
    filter: Box<[f64]>,
    window: VecDeque<f64>,
}

impl InterpolationFilter {
    pub fn new(half_window_len: u32, factor: u8) -> Self {
        let mut filter: Box<[f64]> =
            lowpass_fir_filter(half_window_len * 2 * u32::from(factor) + 1, 1.0 / f64::from(factor), hamming).collect();
        scale_lowpass_filter(&mut filter);
        let window = VecDeque::with_capacity(filter.len());
        Self { factor, filter, window }
    }

    pub fn window_len(&self) -> usize {
        (self.filter.len() - 1) / usize::from(self.factor)
    }

    pub fn interpolate<'a>(&'a mut self, values: impl IntoIterator<Item = f64> + 'a) -> impl Iterator<Item = f64> + 'a {
        let factor = self.factor as f64;
        let scaled = values.into_iter().map(move |value| value * factor);
        let mut extended = Itertools::intersperse(scaled, 0.0);

        self.window.clear();
        self.window.extend(extended.by_ref().take(self.filter.len() - 1));

        let interpolated = extended.map(move |value| {
            if self.window.len() == self.filter.len() {
                self.window.pop_front();
            }
            self.window.push_back(value);
            zip(self.window.iter().rev(), &*self.filter).map(|(x, y)| x * y).sum()
        });
        interpolated
    }
}
