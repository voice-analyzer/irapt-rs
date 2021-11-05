use core::cell::Cell;

pub trait SetFrom {
    type Item;
    fn set(self, value: Self::Item);
}

macro_rules! impl_for_tuple {
    (($($type:ident),+), ($($index:tt),+)) => {
        impl<$($type: SetFrom),+> SetFrom for ($($type),+) {
            type Item = ($(<$type as SetFrom>::Item),+);
            fn set(self, value: Self::Item) {
                $(
                    self.$index.set(value.$index)
                );+
            }
        }
    }
}

impl_for_tuple!((T, U), (0, 1));
impl_for_tuple!((T, U, V), (0, 1, 2));
impl_for_tuple!((T, U, V, W), (0, 1, 2, 3));

impl<T: Clone> SetFrom for &'_ mut T {
    type Item = T;
    fn set(self, value: T) { *self = value }
}

impl<T: Copy> SetFrom for &'_ Cell<T> {
    type Item = T;
    fn set(self, value: T) { self.set(value) }
}
