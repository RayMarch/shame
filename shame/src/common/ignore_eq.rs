use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InEqOrd<T>(pub T);

#[derive(Debug, Clone, Copy)]
pub struct IgnoreInEqOrdHash<T>(pub T);

impl<T: Display> Display for IgnoreInEqOrdHash<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { <T as Display>::fmt(&self.0, f) }
}

impl<T: Display> Display for InEqOrd<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { <T as Display>::fmt(&self.0, f) }
}

impl<T> From<T> for IgnoreInEqOrdHash<T> {
    fn from(t: T) -> Self { Self(t) }
}

impl<T> From<T> for InEqOrd<T> {
    fn from(t: T) -> Self { Self(t) }
}

impl<T: std::hash::Hash> std::hash::Hash for IgnoreInEqOrdHash<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { 0_u8.hash(state); }
}

impl<T: Ord> Ord for IgnoreInEqOrdHash<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { std::cmp::Ordering::Equal }
}

impl<T: PartialOrd> PartialOrd for IgnoreInEqOrdHash<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(std::cmp::Ordering::Equal) }
}

impl<T: Eq> Eq for IgnoreInEqOrdHash<T> {}

impl<T: PartialEq> PartialEq for IgnoreInEqOrdHash<T> {
    fn eq(&self, other: &Self) -> bool { true }
}

impl<T> std::ops::Deref for IgnoreInEqOrdHash<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> std::ops::DerefMut for IgnoreInEqOrdHash<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T> std::ops::Deref for InEqOrd<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> std::ops::DerefMut for InEqOrd<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}
