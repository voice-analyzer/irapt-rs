#[cfg(feature = "matplotlib-ui")]
pub mod matplotlib;
#[cfg(feature = "plotters-ui")]
pub mod plotters;

use crossbeam_utils::atomic::AtomicCell;
use irapt::EstimatedPitch;

#[derive(Default)]
pub struct DisplayData {
    pitches: AtomicCell<Option<Box<[EstimatedPitch]>>>,
}

impl DisplayData {
    pub fn set_pitches(&self, pitches: Box<[EstimatedPitch]>) {
        self.pitches.store(Some(pitches));
    }
}

#[cfg(feature = "matplotlib-ui")]
pub use self::matplotlib::run;

#[cfg(feature = "plotters-ui")]
pub use self::plotters::run;
