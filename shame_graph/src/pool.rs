use std::{
    cell::{Ref, RefCell, RefMut},
    marker::PhantomData,
    num::NonZeroU32,
    ops::{Deref, DerefMut, Index, IndexMut},
};

pub struct Pool<T> {
    store: RefCell<Vec<T>>,
    generation: NonZeroU32,
}

pub struct PoolRefMut<'a, T> {
    generation: NonZeroU32,
    store: RefMut<'a, Vec<T>>,
}
pub struct PoolRef<'a, T> {
    generation: NonZeroU32,
    store: Ref<'a, Vec<T>>,
}

impl<T> Pool<T> {
    pub fn new(generation: NonZeroU32) -> Pool<T> {
        Pool {
            store: Default::default(),
            generation,
        }
    }

    pub fn borrow_mut(&self) -> PoolRefMut<'_, T> {
        PoolRefMut {
            generation: self.generation,
            store: self.store.borrow_mut(),
        }
    }

    pub fn borrow(&self) -> PoolRef<'_, T> {
        PoolRef {
            generation: self.generation,
            store: self.store.borrow(),
        }
    }
}

impl<T> PoolRefMut<'_, T> {
    pub fn push(&mut self, t: T) -> Key<T> {
        let vec = &mut *self.store;

        let index = vec.len();
        vec.push(t);
        Key::new(index, self.generation)
    }
}

impl<T> Index<Key<T>> for PoolRefMut<'_, T> {
    type Output = T;

    fn index(&self, key: Key<T>) -> &Self::Output {
        assert!(key.generation == self.generation);
        &self.store[key.index]
    }
}

impl<T> IndexMut<Key<T>> for PoolRefMut<'_, T> {
    fn index_mut(&mut self, key: Key<T>) -> &mut Self::Output {
        assert!(key.generation == self.generation);
        &mut self.store[key.index]
    }
}

impl<T> Index<Key<T>> for PoolRef<'_, T> {
    type Output = T;

    fn index(&self, key: Key<T>) -> &Self::Output {
        assert!(key.generation == self.generation);
        &self.store[key.index]
    }
}

impl<'a, T> Deref for PoolRefMut<'a, T> {
    type Target = RefMut<'a, Vec<T>>;

    fn deref(&self) -> &Self::Target { &self.store }
}

impl<'a, T> DerefMut for PoolRefMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.store }
}

impl<'a, T> Deref for PoolRef<'a, T> {
    type Target = Ref<'a, Vec<T>>;

    fn deref(&self) -> &Self::Target { &self.store }
}

#[derive(Eq)]
pub struct Key<T> {
    generation: NonZeroU32,
    pub(super) index: usize,
    phantom: PhantomData<T>,
}

impl<T> std::fmt::Debug for Key<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Key")
            .field("generation", &self.generation)
            .field("index", &self.index)
            .finish()
    }
}

impl<T> PartialEq for Key<T> {
    fn eq(&self, other: &Self) -> bool {
        assert!(self.generation == other.generation);
        self.index == other.index
    }
}

//workaround because we cannot derive copy and clone because of PhantomData
impl<T> Clone for Key<T> {
    fn clone(&self) -> Self {
        Self {
            generation: self.generation,
            index: self.index,
            phantom: PhantomData,
        }
    }
}

impl<T> Copy for Key<T> {}

impl<T> Key<T> {
    fn new(index: usize, generation: NonZeroU32) -> Self {
        Self {
            generation,
            index,
            phantom: PhantomData,
        }
    }

    pub(crate) fn index(&self) -> usize { self.index }
}

impl<T> PoolRef<'_, T> {
    #[allow(unused)]
    pub fn enumerate(&self) -> impl Iterator<Item = (Key<T>, &T)> {
        let generation = self.generation;
        self.iter().enumerate().map(move |(i, t)| (Key::new(i, generation), t))
    }
}

impl<T> PoolRefMut<'_, T> {
    pub fn enumerate(&mut self) -> impl Iterator<Item = (Key<T>, &mut T)> {
        let generation = self.generation;
        self.iter_mut()
            .enumerate()
            .map(move |(i, t)| (Key::new(i, generation), t))
    }
}
