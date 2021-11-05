use core::cell::Cell;

pub struct CopiedCell<I> {
    pub(super) iter: I,
}

impl<'a, T: Copy + 'a, I: Iterator<Item = &'a Cell<T>>> Iterator for CopiedCell<I> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|cell| cell.get())
    }
}
