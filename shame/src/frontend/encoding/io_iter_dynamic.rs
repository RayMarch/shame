use std::{cell::Cell, rc::Rc};

use crate::{
    any::{
        layout::{LayoutCalculator, Repr},
        Any, Attrib, Location, VertexBufferLookupIndex,
    },
    call_info,
    frontend::any::{
        render_io::{self, VertexAttributes, VertexBufferKey},
        InvalidReason,
    },
    ir::{self, pipeline::PipelineError, recording::Context},
    TypeLayout, VertexAttribute, VertexIndex,
};

/// an iterator over the draw command's bound vertex buffers, which also
/// allows random access
///
/// use `.next()` or `.at(...)`/`.index(...)` to access individual vertex buffers
pub struct VertexBufferIterDynamic {
    next_slot: u32,
    location_counter: LocationCounter,
}

impl VertexBufferIterDynamic {
    pub(crate) fn new() -> Self {
        Self {
            next_slot: 0,
            location_counter: LocationCounter::from(0),
        }
    }

    /// access the `i`th vertex buffer
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `gpu_layout`.
    #[track_caller]
    pub(crate) fn at(&mut self, i: u32) -> VertexBufferDynamic {
        self.next_slot = i + 1;
        VertexBufferDynamic::new(i, self.location_counter.clone())
    }

    /// access the next vertex buffer (or the first if no buffer was imported yet)
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `gpu_layout`.
    #[allow(clippy::should_implement_trait)] // not fallible
    #[track_caller]
    pub(crate) fn next(&mut self) -> VertexBufferDynamic {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.at(slot)
    }
}

/// (no documentation - chronicl)
pub struct VertexBufferDynamic {
    slot: VertexBufferKey,
    location_counter: LocationCounter,
}

impl VertexBufferDynamic {
    pub(crate) fn new(slot: u32, location_counter: LocationCounter) -> Self {
        let slot = Any::vertex_buffer_new(slot);
        Self { slot, location_counter }
    }

    /// perform a vertex buffer lookup with a special [`VertexIndex`]
    ///
    /// Vertex buffers are special arrays that can only be indexed **once** by the vertex-index
    /// or by the instance-index associated with every vertex.
    /// (see geometry instancing: https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/).
    ///
    /// after indexing the `VertexBuffer` object is consumed and cannot be used for
    /// another index lookup.
    ///
    /// There are only two [`VertexIndex`] objects in `shame`:
    /// - `draw_context.vertices.index` the per-vertex index as defined by the `sm::Indexing` sequence
    /// - `draw_context.vertices.instance_index` the instance index that each vertex belongs to
    ///
    /// ## Example
    ///
    /// ```
    /// let vb: sm::VertexBufferDynamic = drawcall.vertices.buffers.next_dynamic();
    /// let vertex = vb.at(drawcall.vertices.index);
    /// ```
    ///
    /// see "Fixed Function Vertex Processing" https://docs.vulkan.org/spec/latest/chapters/fxvertex.html
    /// for more information
    #[track_caller]
    pub fn index(self, index: VertexIndex) -> VertexBufferElement { self.at(index) }

    /// perform a vertex buffer lookup with a special [`VertexIndex`]
    ///
    /// Vertex buffers are special arrays that can only be indexed **once** by the vertex-index
    /// or by the instance-index associated with every vertex.
    /// (see geometry instancing: https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/).
    ///
    /// after indexing the `VertexBuffer` object is consumed and cannot be used for
    /// another index lookup.
    ///
    /// There are only two [`VertexIndex`] objects in `shame`:
    /// - `draw_context.vertices.index` the per-vertex index as defined by the `sm::Indexing` sequence
    /// - `draw_context.vertices.instance_index` the instance index that each vertex belongs to
    ///
    /// ## Example
    ///
    /// ```
    /// let vb: sm::VertexBufferDynamic = drawcall.vertices.buffers.next_dynamic();
    /// let vertex = vb.at(drawcall.vertices.index);
    /// ```
    ///
    /// see "Fixed Function Vertex Processing" https://docs.vulkan.org/spec/latest/chapters/fxvertex.html
    /// for more information
    // TODO(chronicl) consider returning VertexAttributeIter immediately here
    #[track_caller]
    pub fn at(self, index: VertexIndex) -> VertexBufferElement {
        VertexBufferElement::new(self.slot, self.location_counter, index.0)
    }
}

/// (no documentation - chronicl)
pub struct VertexBufferElement {
    slot: VertexBufferKey,
    location_counter: LocationCounter,
    lookup: VertexBufferLookupIndex,
}

impl VertexBufferElement {
    pub(crate) fn new(
        slot: VertexBufferKey,
        location_counter: LocationCounter,
        lookup: VertexBufferLookupIndex,
    ) -> Self {
        Self {
            slot,
            location_counter,
            lookup,
        }
    }

    #[track_caller]
    pub(crate) fn anys_from_vertex_attributes(self, attributes: VertexAttributes) -> Vec<Any> {
        Any::vertex_buffer_extend(
            &self.slot,
            self.lookup,
            attributes.stride,
            attributes
                .attribs
                .into_iter()
                .map(|attr| (self.location_counter.next(), attr)),
        )
    }

    /// (no documentation - chronicl)
    #[track_caller]
    pub fn attribute_iter(self, repr: Repr) -> VertexAttributeIter {
        VertexAttributeIter::new(self.slot, self.location_counter, self.lookup, repr)
    }
}

/// (no documentation - chronicl)
pub struct VertexAttributeIter {
    slot: VertexBufferKey,
    location_counter: LocationCounter,
    lookup: VertexBufferLookupIndex,
    layout_calculator: LayoutCalculator,
    repr: Repr,
}

impl VertexAttributeIter {
    fn new(
        slot: VertexBufferKey,
        location_counter: LocationCounter,
        lookup: VertexBufferLookupIndex,
        repr: Repr,
    ) -> Self {
        Self {
            slot,
            location_counter,
            lookup,
            layout_calculator: LayoutCalculator::new(repr),
            repr,
        }
    }


    // TODO(chronicl) consider methods that allow custom_min_align and custom_min_size
    /// (no documentation - chronicl)
    #[allow(clippy::should_implement_trait)]
    #[track_caller]
    pub fn next<T: VertexAttribute>(&mut self) -> T { self.at(self.location_counter.next().0) }

    /// (no documentation - chronicl)
    #[track_caller]
    pub fn at<T: VertexAttribute>(&mut self, location: u32) -> T {
        let format = T::vertex_attrib_format();
        let ty = T::layoutable_type_sized();
        let location = Location(location);
        let (size, align) = ty.byte_size_and_align(self.repr);
        let offset = self.layout_calculator.extend(size, align, None, None, false);
        let stride = self.layout_calculator.array_element_stride();
        let attribute = render_io::VertexAttribute { offset, format };

        let anys = Any::vertex_buffer_extend(&self.slot, self.lookup, stride, [(location, attribute)]);
        T::from_anys(anys.into_iter())
    }
}


/// (no documentation - chronicl)
#[derive(Clone)]
pub struct LocationCounter(Rc<Cell<u32>>);

impl LocationCounter {
    pub(crate) fn next(&self) -> Location {
        let i = self.0.get();
        self.0.set(i + 1);
        Location(i)
    }

    /// Adds `n` to the location counter and returns the first location
    /// of the range the add covers.
    pub(crate) fn set(&self, n: u32) -> Location {
        let i = self.0.get();
        self.0.set(i + n);
        Location(i)
    }
}

impl From<u32> for LocationCounter {
    fn from(value: u32) -> Self { Self(Rc::new(Cell::new(value))) }
}
