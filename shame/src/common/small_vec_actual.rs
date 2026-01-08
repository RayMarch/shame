#![allow(missing_docs)]
use std::iter::once;
use std::ops::{Deref, DerefMut};

use crate::common::integer::post_inc_u32;

use super::integer::post_inc_usize;

/// just a short "smallvec" implementation to avoid another dependency
/// on this crate
///
/// the small_vec_dummy module is there to test if this is actually faster.
/// So far this `SmallVec` seems to be a bit faster on release builds.
#[derive(Debug, Clone)]
pub struct SmallVec<T, const N: usize>(Inner<T, N>);

#[derive(Debug, Clone)]
enum Inner<T, const N: usize> {
    Empty,
    Local { len: usize, store: [T; N] },
    Vec(Vec<T>),
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self { SmallVec(Inner::Empty) }
}

impl<T, const N: usize> Deref for SmallVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            Inner::Empty => &[],
            Inner::Local { len, store } => &store[0..*len],
            Inner::Vec(vec) => vec.as_slice(),
        }
    }
}

impl<T, const N: usize> DerefMut for SmallVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.0 {
            Inner::Empty => &mut [],
            Inner::Local { len, store } => &mut store[0..*len],
            Inner::Vec(vec) => vec.as_mut_slice(),
        }
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    fn from_iter_fill(iter: impl IntoIterator<Item = T>, mut produce: impl FnMut() -> T) -> Self {
        let mut iter = iter.into_iter();
        SmallVec(if iter.size_hint().0 <= N {
            let mut len = 0;
            let store = [(); N].map(|_| match iter.next() {
                None => produce(),
                Some(t) => {
                    len += 1;
                    t
                }
            });
            match iter.next() {
                None => Inner::Local { len, store },
                Some(t) => {
                    //size hint was inaccurate
                    assert!(len == N);
                    let mut vec = Vec::from(store);
                    vec.push(t);
                    vec.extend(iter);
                    Inner::Vec(vec)
                }
            }
        } else {
            Inner::Vec(iter.collect())
        })
    }

    pub fn clear(&mut self) {
        match &mut self.0 {
            Inner::Empty => (),
            Inner::Local { .. } => self.0 = Inner::Empty,
            Inner::Vec(v) => v.clear(),
        }
    }

    /// tries to collect `Ok` variants from `iter` but stops once the first `Err`
    /// happens and returns it
    pub fn try_from_iter<E>(iter: impl IntoIterator<Item = Result<T, E>>) -> Result<Self, E>
    where
        T: Clone,
    {
        let mut vec = Self::default();
        for result in iter {
            vec.push(result?)
        }
        Ok(vec)
    }

    /// tries to collect `Some` variants from `iter` but if it encounters any
    /// `None` it returns `None` itself.
    pub fn from_opt_iter(iter: impl IntoIterator<Item = Option<T>>) -> Option<Self>
    where
        T: Clone,
    {
        let mut vec = Self::default();
        for result in iter {
            vec.push(result?)
        }
        Some(vec)
    }

    pub fn push(&mut self, t: T)
    where
        T: Clone,
    {
        let new_variant: Option<Inner<T, N>> = match &mut self.0 {
            Inner::Empty => Some(match N {
                0 => Inner::Vec(vec![t]),
                _ => Inner::Local {
                    store: [(); N].map(|_| t.clone()),
                    len: 1,
                },
            }),
            Inner::Local { len, store } => {
                match store.get_mut(*len) {
                    Some(place) => {
                        *place = t;
                        *len += 1;
                        None
                    }
                    None => {
                        // store is full
                        let mut vec = Vec::with_capacity(N + 1);
                        vec.extend(store.iter().cloned());
                        vec.push(t);
                        Some(Inner::Vec(vec))
                    }
                }
            }
            Inner::Vec(v) => {
                v.push(t);
                None
            }
        };
        if let Some(new_variant) = new_variant {
            self.0 = new_variant
        }
    }
}

impl<T: Clone, const N: usize> FromIterator<T> for SmallVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        match iter.next() {
            None => Self::default(),
            Some(first) => {
                let re_iter = once(first.clone()).chain(iter);
                Self::from_iter_fill(re_iter, || first.clone())
            }
        }
    }
}

impl<T: Copy, const N: usize, const M: usize> TryFrom<SmallVec<T, M>> for [T; N] {
    type Error = SmallVec<T, M>;

    fn try_from(small_vec: SmallVec<T, M>) -> Result<Self, Self::Error> {
        match small_vec.0 {
            Inner::Empty => match <[T; N]>::try_from(Vec::new()) {
                Ok(x) => Ok(x),
                Err(e) => Err(Inner::Vec(e)),
            },
            Inner::Local { len, store } if len == N => match store.first() {
                Some(first) => {
                    let mut array = [*first; N];
                    array[1..N].copy_from_slice(&store[1..N]);
                    Ok(array)
                }
                None => Err(Inner::Empty), //unreachable
            },
            Inner::Vec(vec) => match <[T; N]>::try_from(vec) {
                Ok(x) => Ok(x),
                Err(e) => Err(Inner::Vec(e)),
            },
            _ => Err(small_vec.0),
        }
        .map_err(|e| SmallVec(e))
    }
}

impl<T, const N: usize> Eq for SmallVec<T, N> where T: Eq {}

impl<T, const N: usize> PartialEq for SmallVec<T, N>
where
    T: PartialEq<T>,
{
    fn eq(&self, other: &Self) -> bool { self.iter().zip(other.iter()).all(|(a, b)| a == b) }
}

impl<T: Clone, const N: usize> From<&[T]> for SmallVec<T, N> {
    fn from(slice: &[T]) -> Self {
        Self(match slice.len() {
            0 => Inner::Empty,
            len if len < N => Inner::Local {
                len,
                store: {
                    let mut i = 0;
                    [(); N].map(|_| {
                        slice
                            .get(post_inc_usize(&mut i))
                            .cloned()
                            .unwrap_or_else(|| slice[0].clone())
                    })
                },
            },
            len => Inner::Vec(slice.into()),
        })
    }
}

impl<T, const N: usize> std::borrow::Borrow<[T]> for SmallVec<T, N> {
    fn borrow(&self) -> &[T] { self }
}

impl<T, const N: usize> std::fmt::Display for SmallVec<T, N>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut iter = self.iter();
        if let Some(first) = iter.next() {
            write!(f, "{}", first)?;
            for item in iter {
                write!(f, ", {}", item)?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inaccurate_size_hint() {
        let v = SmallVec::<_, 3>::from_iter((0..).take_while(|k| *k < 5));
        assert_eq!(&*v, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn smaller_than_local_store() {
        let v = SmallVec::<_, 5>::from_iter((0..).take_while(|k| *k < 3));
        assert_eq!(&*v, &[0, 1, 2]);
    }

    #[test]
    fn bigger_than_local_store() {
        let v = SmallVec::<_, 5>::from_iter((0..).take_while(|k| *k < 8));
        assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn accurate_size_hint() {
        let v = SmallVec::<_, 3>::from_iter(0..5);
        assert_eq!(&*v, &[0, 1, 2, 3, 4]);

        let v = SmallVec::<_, 8>::from_iter(0..5);
        assert_eq!(&*v, &[0, 1, 2, 3, 4]);

        let v: SmallVec<_, 8> = (0..10).collect();
        assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let v: SmallVec<_, 8> = (0..8).collect();
        assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 6, 7]);

        let v: SmallVec<_, 8> = (0..0).collect();
        assert_eq!(&*v, &[]);

        let v = SmallVec::<_, 8>::from_iter(0..0);
        assert_eq!(&*v, &[]);
    }

    #[test]
    fn small_vec_from_slice() {
        for n in 0..16 {
            let vec: Vec<u32> = (0..n).collect();
            let slice: &[u32] = &vec;
            let smallvec = SmallVec::<u32, 4>::from(slice);
            let smallvec_slice: &[u32] = &smallvec;
            assert_eq!(&smallvec_slice, &slice);
        }
    }

    #[test]
    fn small_vec_from_iter() {
        for n in 0..16 {
            let vec: Vec<u32> = (0..n).collect();
            let slice: &[u32] = &vec;
            let smallvec: SmallVec<u32, 4> = vec.clone().into_iter().collect();
            let smallvec_slice: &[u32] = &smallvec;
            assert_eq!(&smallvec_slice, &slice);
        }
    }
}
