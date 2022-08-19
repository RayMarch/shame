use std::ops::Range;

pub struct LocationIter<I: Iterator<Item = u32>> {
    iter: I,
    min_valid_next: u32, //the minimum value the next returned element has to have
}

impl<I: Iterator<Item = u32>> LocationIter<I> {
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            min_valid_next: u32::MIN,
        }
    }

    pub fn next(&mut self, width: u32) -> Option<Range<u32>> {
        assert!(
            width <= 4,
            "there is no tensor type with a vertex attribute width of {} which is > 4",
            width
        );
        let timeout = 1024;
        for _ in 0..timeout {
            match self.iter.next() {
                Some(loc) => {
                    if loc >= self.min_valid_next {
                        self.min_valid_next = loc + width;
                        return Some(loc..loc + width);
                    }
                }
                None => return None,
            }
        }
        panic!("vertex attribute location iterator has not found a fitting vertex attribute slot of width {} after {} iterations", width, timeout);
    }
}
