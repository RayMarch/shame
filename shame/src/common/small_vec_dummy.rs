use std::ops::Deref;

/// dummy smallvec implementation that uses a regular vec underneath to do AB tests with
#[derive(Debug, Clone)]
pub struct SmallVec<T, const N: usize>(Vec<T>);

impl<T, const N: usize> Deref for SmallVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T, const N: usize> SmallVec<T, N> {
    /// tries to collect `Some` variants from `iter` but if it encounters any
    /// `None` it returns `None` itself.
    pub fn from_opt_iter(iter: impl IntoIterator<Item = Option<T>>) -> Option<Self>
    where
        T: Clone,
    {
        let mut vec = Vec::with_capacity(N);
        for result in iter {
            vec.push(result?)
        }
        Some(Self(vec))
    }
}

impl<T, const N: usize> FromIterator<T> for SmallVec<T, N> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self { Self(Vec::from_iter(iter)) }
}
