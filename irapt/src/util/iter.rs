pub mod copied_cell;
pub mod set_from;

use self::copied_cell::CopiedCell;
use self::set_from::SetFrom;

pub trait IteratorExt: Iterator {
    fn copied_cell(self) -> CopiedCell<Self> where Self: Sized {
        CopiedCell { iter: self }
    }

    #[inline]
    fn set_from<'a, T, SetFromT, I>(&mut self, from: I) -> usize
    where Self: Iterator<Item = SetFromT>,
          I: IntoIterator<Item = T>,
          SetFromT: SetFrom<Item = T>,
    {
        let mut count = 0;
        for value in from {
            match self.next() {
                Some(place) => place.set(value),
                None => break,
            }
            count += 1;
        }
        count
    }
}

impl<I: Iterator> IteratorExt for I {}
