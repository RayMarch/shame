use std::{cell::Cell, num::NonZeroU32};

pub trait CellNonZeroU32Ext {
    fn increment_by(&self, amount: u32) -> NonZeroU32;
}

impl CellNonZeroU32Ext for Cell<NonZeroU32> {
    fn increment_by(&self, amount: u32) -> NonZeroU32 {
        let next: u32 = self.get().into();
        let next = NonZeroU32::new(next + amount).unwrap();
        self.set(next);
        next
    }
}

impl<I: Iterator> IteratorExt for I {}
pub trait IteratorExt: Iterator where Self: Sized {
    
    ///applies f to all elements in the iterator, if f's return values 
    ///all compare equal to the first one, returns Some(f(first)), otherwise None.
    ///Returns None also for empty iterators.
    fn all_same<R: Eq>(self, f: impl FnMut(Self::Item) -> R) -> Option<R> {
        let mut iter = self.map(f);
        iter.next().and_then(|first| iter.all(|i| i == first).then(|| first))
    }

    fn all_unique(self) -> bool 
    where Self: Clone, Self::Item: Eq {
        self.all_unique_by(|x, y| x == y)
    }

    ///does not require is_same to be transitive
    fn all_unique_by(mut self, mut is_same: impl FnMut(&Self::Item, &Self::Item) -> bool) -> bool
    where Self: Clone {
        loop {match self.next() {
            Some(x) => if self.clone().any(|y| is_same(&x, &y)) {break false;}
            _ => break true
        }}
    }

    fn find_pair<R>(mut self, mut f: impl FnMut(&Self::Item, &Self::Item) -> Option<R>) -> Option<R>
    where Self: Clone {
        while let Some(x) = self.next() {
            for y in self.clone() {
                if let Some(r) = f(&x, &y) {
                    return Some(r)
                }
            }
        }
        None
    }

}

pub enum Either<L, R> {
    Left(L),
    Right(R),
}

pub struct Deferred<F: FnOnce()>(Option<F>);

pub fn defer(f: impl FnOnce()) -> impl Drop {
    Deferred(Some(f))
}

impl<F: FnOnce()> Drop for Deferred<F> {
    fn drop(&mut self) {self.0.take().unwrap()()}
}

#[macro_export]
macro_rules! unwrap_variant {
    ($value: expr, $pattern: pat => $unwrapped_value: expr) => {
        match $value {
            $pattern => $unwrapped_value,
            _ => panic!("pattern '{}' doesn't match in unwrap_variant", stringify!($pattern)),
        }
    };
}

pub fn ranges_overlap<T>(a: &std::ops::Range<T>, b: &std::ops::Range<T>) -> bool
where T: Into<i64> + Clone {
    let (a, b): ((i64, i64), (i64, i64)) = ((a.start.clone().into(), a.end.clone().into()), (b.start.clone().into(), b.end.clone().into()));
    a.0.max(b.0) <= (a.1-1).min(b.1-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert!(ranges_overlap(&(0..1), &(0..1)));
        assert!(ranges_overlap(&(0..2), &(1..3)));
        assert!(ranges_overlap(&(1..3), &(0..2)));

        assert!(!ranges_overlap(&(0..1), &(1..2)));
        assert!(!ranges_overlap(&(1..2), &(0..1)));
        assert!(!ranges_overlap(&(2..3), &(0..1)));
        assert!(!ranges_overlap(&(-1..0), &(0..4)));
        assert!(!ranges_overlap(&(0..4), &(-1..0)));
    }
}

#[derive(Clone)]
pub struct StartIterFrom<T: Copy, F: FnMut(T) -> Option<T>> {
    first: Option<T>,
    current: Option<T>,
    next_fn: F,
}

impl<T: Copy, F: FnMut(T) -> Option<T>> Iterator for StartIterFrom<T, F> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.first {
            Some(t) => {
                self.first = None;
                Some(t)
            }
            None => {
                self.current = self.current.and_then(|curr| (self.next_fn)(curr));
                self.current
            }
        }
    }
}

pub fn start_iter_from<T: Copy, F>(from: Option<T>, next_fn: F) -> StartIterFrom<T, F>
where F: FnMut(T) -> Option<T> {
    StartIterFrom {
        first: from,
        current: from, 
        next_fn 
    }
}

pub fn new_array_enumerate<T, const N: usize>(mut f: impl FnMut(usize) -> T) -> [T; N] {
    let mut i = 0..;
    [(); N].map(|_| f(i.next().unwrap()))
}