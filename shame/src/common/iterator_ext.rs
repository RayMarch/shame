///  if all elements are `Some` returns `it.filter_map().collect()`, otherwise `None`
pub fn try_collect<T, It: Iterator<Item = Option<T>>>(it: It) -> Option<Box<[T]>> {
    let mut i = 0;
    let result: Box<[T]> = it
        .filter_map(|item| {
            i += 1;
            item
        })
        .collect();
    (result.len() == i).then_some(result)
}

impl<I: Iterator> IteratorExt for I {}
pub trait IteratorExt: Iterator
where
    Self: Sized,
{
    fn all_unique(self) -> bool
    where
        Self: Clone,
        Self::Item: Eq,
    {
        self.all_unique_by(|x, y| x == y)
    }

    /// does not require is_same to be transitive
    fn all_unique_by(mut self, mut is_same: impl FnMut(&Self::Item, &Self::Item) -> bool) -> bool
    where
        Self: Clone,
    {
        loop {
            match self.next() {
                Some(x) => {
                    if self.clone().any(|y| is_same(&x, &y)) {
                        break false;
                    }
                }
                _ => break true,
            }
        }
    }
}
