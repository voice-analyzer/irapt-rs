use alloc::collections::VecDeque;

#[derive(Default)]
pub struct InputBufferCursors<T> {
    cursors: T,
    index: usize,
}

#[derive(Default)]
pub struct InputBufferCursor {
    index: usize,
}

impl<T> InputBufferCursors<T> {
    pub fn new(cursors: T) -> Self {
        Self { cursors, index: 0 }
    }
}

impl<T> InputBufferCursors<T> {
    pub fn cursors_mut(&mut self) -> &mut T {
        &mut self.cursors
    }
}

impl<T> InputBufferCursors<T>
where for <'a> &'a mut T: IntoIterator<Item = &'a mut InputBufferCursor>,
{
    pub fn advance_buffer<U>(&mut self, buffer: &mut VecDeque<U>) {
        let min_cursor = (&mut self.cursors).into_iter().min_by_key(|cursor| cursor.index);
        let min_cursor_index = min_cursor.map(|cursor| cursor.index).unwrap_or(buffer.len());
        if let Some(drain_length) = min_cursor_index.checked_sub(self.index) {
            buffer.drain(..drain_length);
            for cursor in &mut self.cursors {
                cursor.index -= drain_length;
            }
        }
    }
}

impl InputBufferCursor {
    pub fn advance(&mut self, amount: usize) {
        self.index += amount;
    }

    pub fn index(&self) -> usize {
        self.index
    }
}
