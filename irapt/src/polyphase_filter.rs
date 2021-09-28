use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use core::cell::Cell;
use itertools::{chain, repeat_n, zip};
use num::Complex;
use rustfft::{Fft, FftPlanner};

use crate::fir_filter::{hamming, lowpass_fir_filter};

pub struct PolyphaseFilter {
    filter:       Box<[f64]>,
    ifft:         Arc<dyn Fft<f64>>,
    ifft_scratch: Box<[Complex<f64>]>,
}

impl PolyphaseFilter {
    pub fn new(window_len: u32, channel_count: usize) -> Self {
        let filter = lowpass_fir_filter(window_len + 1, 1.0 / channel_count as f64, hamming).collect();
        let ifft = FftPlanner::new().plan_fft_inverse(channel_count);
        let ifft_scratch = vec![<_>::default(); ifft.get_inplace_scratch_len()].into();
        Self {
            filter,
            ifft,
            ifft_scratch,
        }
    }

    pub fn process<I>(&mut self, samples: I, channels: &mut [Complex<f64>])
    where I: IntoIterator<Item = f64> {
        let filter_pad_len = (channels.len() * 2 - self.filter.len() + 1) / 2;
        let filter = chain!(repeat_n(&0.0, filter_pad_len), &*self.filter);
        let filtered_samples = zip(samples, filter).map(|(sample, filter)| sample * filter);

        channels.fill(Complex::new(0.0, 0.0));
        let channels_cells = Cell::from_mut(channels).as_slice_of_cells();
        let channels_cyclical_iter = channels_cells.iter().cycle();
        zip(channels_cyclical_iter, filtered_samples).for_each(|(channel, filtered_sample)| channel.set(channel.get() + filtered_sample));

        self.ifft.process_with_scratch(channels, &mut self.ifft_scratch);
    }
}

#[cfg(test)]
mod test;
