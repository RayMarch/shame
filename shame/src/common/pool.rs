use std::{
    any::Any,
    cell::{Ref, RefCell, RefMut},
    fmt::{Display, Pointer},
    marker::PhantomData,
    num::NonZeroU32,
    ops::{Deref, DerefMut, Index, IndexMut, Range},
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

    #[track_caller]
    pub fn borrow_mut(&self) -> PoolRefMut<'_, T> {
        PoolRefMut {
            generation: self.generation,
            store: self.store.borrow_mut(),
        }
    }

    #[track_caller]
    pub fn borrow(&self) -> PoolRef<'_, T> {
        PoolRef {
            generation: self.generation,
            store: self.store.borrow(),
        }
    }

    #[track_caller]
    pub fn try_borrow(&self) -> Option<PoolRef<'_, T>> {
        Some(PoolRef {
            generation: self.generation,
            store: self.store.try_borrow().ok()?,
        })
    }
}

impl<T> PoolRefMut<'_, T> {
    pub fn push(&mut self, t: T) -> Key<T> {
        let vec = &mut *self.store;

        let index = vec.len();
        vec.push(t);
        Key::new(index as u32, self.generation)
    }

    pub fn generation(&self) -> NonZeroU32 { self.generation }

    pub fn get_mut(&mut self, key: Key<T>) -> Option<&mut T> {
        (key.generation == self.generation)
            .then_some(())
            .and_then(|_| self.store.get_mut(key.index as usize))
    }

    pub fn get(&self, key: Key<T>) -> Option<&T> {
        (key.generation == self.generation)
            .then_some(())
            .and_then(|_| self.store.get(key.index as usize))
    }
}

impl<T> Index<Key<T>> for PoolRefMut<'_, T> {
    type Output = T;

    fn index(&self, key: Key<T>) -> &Self::Output {
        assert!(key.generation == self.generation);
        &self.store[key.index as usize]
    }
}

impl<T> IndexMut<Key<T>> for PoolRefMut<'_, T> {
    fn index_mut(&mut self, key: Key<T>) -> &mut Self::Output {
        assert!(key.generation == self.generation);
        &mut self.store[key.index as usize]
    }
}

impl<T> Index<Key<T>> for PoolRef<'_, T> {
    type Output = T;

    fn index(&self, key: Key<T>) -> &Self::Output {
        assert!(key.generation == self.generation);
        &self.store[key.index as usize]
    }
}

impl<'a, T> Deref for PoolRefMut<'a, T> {
    type Target = RefMut<'a, Vec<T>>;

    fn deref(&self) -> &Self::Target { &self.store }
}

impl<T> DerefMut for PoolRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.store }
}

impl<'a, T> Deref for PoolRef<'a, T> {
    type Target = Ref<'a, Vec<T>>;

    fn deref(&self) -> &Self::Target { &self.store }
}

pub struct Key<T> {
    generation: NonZeroU32,
    pub(super) index: u32,
    phantom: PhantomData<T>,
}

impl<T> Key<T> {
    pub fn generation(&self) -> NonZeroU32 { self.generation }
}

impl<T> std::fmt::Debug for Key<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Key")
            .field("generation", &self.generation)
            .field("index", &self.index)
            .finish()
    }
}

impl<T> Eq for Key<T> {}

impl<T> PartialEq for Key<T> {
    fn eq(&self, other: &Self) -> bool {
        assert!(self.generation == other.generation);
        self.index == other.index
    }
}

//workaround because we cannot derive copy and clone because of PhantomData
impl<T> Clone for Key<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> Copy for Key<T> {}

impl<T> Key<T> {
    fn new(index: u32, generation: NonZeroU32) -> Self {
        Self {
            generation,
            index,
            phantom: PhantomData,
        }
    }

    pub(crate) fn index(&self) -> usize { self.index as usize }
}

impl<T> PoolRef<'_, T> {
    #[allow(unused)]
    pub fn enumerate(&self) -> impl DoubleEndedIterator<Item = (Key<T>, &T)> {
        let generation = self.generation;
        self.iter()
            .enumerate()
            .map(move |(i, t)| (Key::new(i as u32, generation), t))
    }

    pub fn generation(&self) -> NonZeroU32 { self.generation }

    pub fn lookup_fn<'a>(&'a self) -> impl FnMut(Key<T>) -> &'a T + Copy { move |key| &self[key] }

    pub fn get(&self, key: Key<T>) -> Option<&T> {
        (key.generation == self.generation)
            .then_some(())
            .and_then(|_| self.store.get(key.index as usize))
    }

    pub fn keys(&self) -> KeyRange<T> {
        KeyRange {
            index_range: 0..(self.len() as u32),
            generation: self.generation,
            phantom: PhantomData,
        }
    }
}

impl<T> PoolRefMut<'_, T> {
    pub fn enumerate_mut(&mut self) -> impl Iterator<Item = (Key<T>, &mut T)> {
        let generation = self.generation;
        self.iter_mut()
            .enumerate()
            .map(move |(i, t)| (Key::new(i as u32, generation), t))
    }

    pub fn enumerate(&self) -> impl Iterator<Item = (Key<T>, &T)> {
        let generation = self.generation;
        self.iter()
            .enumerate()
            .map(move |(i, t)| (Key::new(i as u32, generation), t))
    }

    pub fn keys(&self) -> KeyRange<T> {
        KeyRange {
            index_range: 0..(self.len() as u32),
            generation: self.generation,
            phantom: PhantomData,
        }
    }
}

pub struct KeyRange<T> {
    index_range: Range<u32>,
    generation: NonZeroU32,
    phantom: PhantomData<T>,
}

impl<T> Iterator for KeyRange<T> {
    type Item = Key<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_range.next().map(|index| Key {
            generation: self.generation,
            index,
            phantom: PhantomData,
        })
    }
}

impl<T> DoubleEndedIterator for KeyRange<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.index_range.next_back().map(|index| Key {
            generation: self.generation,
            index,
            phantom: PhantomData,
        })
    }
}
