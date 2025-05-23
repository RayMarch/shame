use std::{fmt::Display, rc::Rc};

use thiserror::Error;

use crate::__private::SmallVec;
use crate::frontend::any::Any;
use crate::frontend::rust_types::type_layout::{layoutable, TypeLayout};
use crate::{ir};
use crate::{
    call_info,
    common::iterator_ext::try_collect,
    frontend::{
        encoding::{
            fill::{Fill, PickVertex},
            EncodingErrorKind,
        },
        rust_types::type_layout::TypeLayoutSemantics,
    },
    ir::{
        expr::{BuiltinShaderIn, BuiltinShaderIo, Expr, Interpolator, ShaderIo},
        ir_type::{stride_of_array_from_element_align_size, CanonName, LenEven, TextureFormatId},
        pipeline::{PipelineError, RecordedWithIndex},
        recording::Context,
        Len, PackedVector, ScalarType, SizedType, StoreType, TextureFormatWrapper, Type,
    },
};

use super::{blend::Blend, record_node, InvalidReason};

#[allow(missing_docs)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location(pub u32);

impl Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

impl std::ops::Deref for Location {
    type Target = u32;

    fn deref(&self) -> &Self::Target { &self.0 }
}

#[derive(Debug, Error, Clone)]
#[allow(missing_docs)]
pub enum VertexLayoutError {
    #[error(
        "the provided vertex attribute location iterator ran out of locations. It is recommended to provide an endless iterator, for example `0..`."
    )]
    LocationIteratorRanOutOfLocations,
    // #[error("field `{0}` of type `{1}` cannot be part of a vertex buffer. Only scalar and vector types allowed.")]
    // FieldCannotBeVertexAttribute(CanonName, Type),
}

#[derive(Debug, Clone)]
/// (no documentation - chronicl)
pub struct VertexAttributes {
    /// (no documentation - chronicl)
    pub stride: u64,
    /// (no documentation - chronicl)
    pub attribs: Box<[VertexAttribute]>,
}

#[derive(Debug, Clone, Copy)]
/// Vertex Attribute information - offset and format.
pub struct VertexAttribute {
    /// (no documentation - chronicl)
    pub offset: u64,
    /// (no documentation - chronicl)
    pub format: VertexAttribFormat,
}

/// location and format of a vertex attribute
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Attrib {
    /// the byte-offset of the first occurence of this vertex attribute within the vertex buffer
    pub offset: u64,
    /// the vertex attribute location, which is used to associate the information of this struct
    /// with the vertex-shader input values inside the vertex shader code.
    pub location: Location,
    /// the datatype of this vertex attribute
    pub format: VertexAttribFormat,
}

impl Attrib {
    #[allow(missing_docs)]
    pub fn new(offset: u64, location: Location, format: VertexAttribFormat) -> Self {
        Self {
            offset,
            location,
            format,
        }
    }
}

/// The index that is used to look up the vertex attributes within a vertex buffer
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexBufferLookupIndex {
    /// the individual values of the buffer are looked up according to the
    /// indexing in the index buffer. If no index buffer is used, each value
    /// up to the index limit is looked up by one vertex.
    #[default]
    VertexIndex,
    /// the `i`th value within the buffer is looked up by every vertex of the
    /// `i`th instance.
    InstanceIndex,
}

impl Any {
    /// u32x1 vertex id, can only be created once per pipeline
    #[track_caller]
    pub fn new_vertex_id() -> Any {
        let io = BuiltinShaderIo::Get(BuiltinShaderIn::VertexIndex);
        record_node(call_info!(), Expr::ShaderIo(ShaderIo::Builtin(io)), &[])
    }

    /// u32x1 instance id, can only be created once per pipeline
    #[track_caller]
    pub fn new_instance_id() -> Any {
        let io = BuiltinShaderIo::Get(BuiltinShaderIn::InstanceIndex);
        record_node(call_info!(), Expr::ShaderIo(ShaderIo::Builtin(io)), &[])
    }
}

/// the datatype of a vertex attribute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexAttribFormat {
    /// regular [`crate::vec`] types without bools
    Fine(layoutable::Vector),
    /// packed [`crate::packed::PackedVec`] types
    Coarse(PackedVector),
}

/// The memory layout and lookup method of a vertex buffer.
///
/// A vertex buffer can consist of multiple vertex attributes, each of which
/// are interleaved inside the buffer and repeated the same amount of times.
///
/// Each vertex attribute is of [`vec`] or [`packed::PackedVec`] type.
///
/// see https://docs.rs/wgpu/latest/wgpu/struct.VertexBufferLayout.html
///
/// or https://www.w3.org/TR/webgpu/#dictdef-gpuvertexbufferlayout
///
/// [`vec`]: crate::vec
/// [`packed::PackedVec`]: crate::packed::PackedVec
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexBufferLayoutRecorded {
    /// Slot of the layout
    pub slot: u32,
    /// The index that is used to look up the vertex attributes within a vertex buffer
    ///
    /// either `vertex_index` or `instance_index`
    pub lookup: VertexBufferLookupIndex,
    /// The amount of bytes between the first occurence of a vertex attribute in
    /// the buffer and the next occurence of that same attribute in the buffer.
    pub stride: u64,
    /// Location and layout information of each vertex attribute
    pub attribs: Vec<RecordedWithIndex<Attrib>>,
}

/// a mask that specifies which color components should be written to and which
/// ones should be ignored.
///
/// - components assigned `false` will not be written to
/// - components assigned `true` will be written to
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChannelWrites {
    /// whether to write to the red channel
    pub r: bool,
    /// whether to write to the green channel, if present
    pub g: bool,
    /// whether to write to the blue channel, if present
    pub b: bool,
    /// whether to write to the alpha channel, if present
    pub a: bool,
}

impl ChannelWrites {
    /// construct channel writes with an array of digits, where nonzero digits mean
    /// that channel is being written to.
    ///
    /// example:
    /// ```
    /// let c = ChannelWrite::from_digits([1, 1, 0, 1])
    /// //is equivalent to
    /// let c = ChannelWrites {
    ///     r: true,
    ///     g: true,
    ///     b: false,
    ///     a: true,
    /// }
    /// ```
    pub fn from_digits(digits: [u8; 4]) -> Self { Self::from_bools(digits.map(|d| d != 0)) }

    /// construct channel writes with an array of `bool`s, `true` means
    /// that channel is being written to.
    ///
    /// example:
    /// ```
    /// let c = ChannelWrite::from_digits([1, 1, 0, 1])
    /// //is equivalent to
    /// let c = ChannelWrites {
    ///     r: true,
    ///     g: true,
    ///     b: false,
    ///     a: true,
    /// }
    /// ```
    pub fn from_bools([r, g, b, a]: [bool; 4]) -> Self { Self { r, g, b, a } }

    /// A mask that writes to all channels
    pub fn rgba() -> Self { [1; 4].into() }

    /// A mask that writes to no channels
    pub fn empty() -> Self { [0; 4].into() }
}

impl From<[u8; 4]> for ChannelWrites {
    fn from(digits: [u8; 4]) -> Self { ChannelWrites::from_digits(digits) }
}

impl From<[bool; 4]> for ChannelWrites {
    fn from(bools: [bool; 4]) -> Self { ChannelWrites::from_bools(bools) }
}

impl Default for ChannelWrites {
    fn default() -> Self {
        Self {
            r: true,
            g: true,
            b: true,
            a: true,
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: TextureFormatWrapper,
    pub blend: Option<Blend>,
    pub write_mask: ChannelWrites,
}

impl ColorTarget {
    #[allow(missing_docs)]
    pub fn new<F: TextureFormatId>(format: F, blend: Option<Blend>, write_mask: ChannelWrites) -> Self {
        Self {
            format: TextureFormatWrapper::new(format),
            blend,
            write_mask,
        }
    }
}

impl VertexAttribFormat {
    #[allow(missing_docs)]
    pub fn type_in_shader(self) -> SizedType {
        match self {
            VertexAttribFormat::Fine(v) => SizedType::Vector(v.len, v.scalar.into()),
            VertexAttribFormat::Coarse(coarse) => coarse.decompressed_ty(),
        }
    }
}

/// Signifies ownership of a recorded vertex buffer layout if `VertexBufferKey::is_valid` is true.
///
/// Can be used to extend a recorded vertex buffer layout via `Any::vertex_buffer_extend`.
/// If the key is not valid the resulting `Any`s will not be valid, but
/// `Any::vertex_buffer_extend` is still okay to call.
pub struct VertexBufferAny(Result<usize, InvalidReason>);

impl VertexBufferAny {
    /// Returns whether this key is valid.
    pub fn is_valid(&self) -> bool { self.0.is_ok() }
}

impl Any {
    /// Obtains ownership over a new vertex buffer layout slot. The `VertexBufferKey` can
    /// be used with `Any::vertex_buffer_extend` to add attributes to the vertex buffer layout at the slot.
    #[track_caller]
    pub fn vertex_buffer_new(slot: u32) -> VertexBufferAny {
        let result = Context::try_with(call_info!(), |ctx| {
            ctx.push_error_if_outside_encoding_scope("vertex buffer import");

            let buffers = &mut ctx.render_pipeline_mut().vertex_buffers;

            match buffers.iter().find(|b| b.slot == slot) {
                Some(existing) => {
                    ctx.push_error(PipelineError::DuplicateVertexBufferImport(slot, existing.call_info).into());
                    Err(InvalidReason::ErrorThatWasPushed)
                }
                None => {
                    let i = buffers.len();
                    buffers.push(RecordedWithIndex::new(
                        VertexBufferLayoutRecorded {
                            slot,
                            stride: 1,
                            lookup: VertexBufferLookupIndex::VertexIndex,
                            attribs: Default::default(),
                        },
                        slot,
                        ctx.latest_user_caller(),
                    ));
                    Ok(i)
                }
            }
        })
        .unwrap_or(Err(InvalidReason::CreatedWithNoActiveEncoding));

        VertexBufferAny(result)
    }

    /// Extends the vertex buffer layout at `vertex_buffer_record_index` by the given vertex `attributes`.
    ///
    /// - Later calls to this function extending the same `slot` will overwrite
    ///   the `stride` and `lookup`.
    #[track_caller]
    pub fn vertex_buffer_extend(
        vertex_buffer_record_index: &VertexBufferAny,
        lookup: VertexBufferLookupIndex,
        stride: u64,
        // TODO(chronicl) probably change this to iterator over Attrib
        attributes: impl IntoIterator<Item = (Location, VertexAttribute)>,
    ) -> Vec<Any> {
        let call_info = call_info!();


        let attributes: SmallVec<_, 5> = attributes.into_iter().collect();

        let invalid_anys = |reason| std::iter::repeat_n(Any::new_invalid(reason), attributes.len()).collect();

        Context::try_with(call_info, |ctx| -> Vec<Any> {
            ctx.push_error_if_outside_encoding_scope("vertex attribute import");

            let vertex_buffer_record_index = match vertex_buffer_record_index.0 {
                Ok(i) => i,
                Err(reason) => {
                    match reason {
                        InvalidReason::ErrorThatWasPushed => (),
                        InvalidReason::CreatedWithNoActiveEncoding => {
                            ctx.push_error(PipelineError::VertexBufferCreatedOutsideOfActiveEncoding.into())
                        }
                    }
                    return invalid_anys(InvalidReason::ErrorThatWasPushed);
                }
            };

            let mut rp = ctx.render_pipeline_mut();
            let buffer = &mut rp.vertex_buffers[vertex_buffer_record_index];
            buffer.lookup = lookup;
            buffer.stride = stride;
            let slot = buffer.slot;
            drop(rp);

            let mut anys = Vec::<Any>::new();
            for (location, attr) in attributes.iter() {
                let mut rp = ctx.render_pipeline_mut();
                let mut buffers = &mut rp.vertex_buffers;

                let any = match ensure_location_is_unique(ctx, slot, *location, buffers) {
                    Err(e) => ctx.push_error_get_invalid_any(e.into()),
                    Ok(()) => {
                        buffers[vertex_buffer_record_index].attribs.push(RecordedWithIndex::new(
                            Attrib::new(attr.offset, *location, attr.format),
                            location.0,
                            call_info,
                        ));

                        // Have to drop because record_node needs access.
                        drop(rp);
                        // Order important! must happen after `attribs.push`.
                        record_node(
                            ctx.latest_user_caller(),
                            ShaderIo::GetVertexInput(*location).into(),
                            &[],
                        )
                    }
                };
                anys.push(any);
            }

            anys
        })
        .unwrap_or_else(|| invalid_anys(InvalidReason::CreatedWithNoActiveEncoding))
    }

    #[allow(missing_docs)]
    #[track_caller]
    /// src_color_xn component count must match the target format
    pub fn color_target_write(slot: u32, color_target: ColorTarget, src_color_xn: Any) {
        Context::try_with(call_info!(), |ctx| {
            ctx.push_error_if_outside_encoding_scope("writing to color targets");
            ctx.render_pipeline_mut().color_targets.push(RecordedWithIndex::new(
                color_target,
                slot,
                ctx.latest_user_caller(),
            ));
            record_node(
                ctx.latest_user_caller(),
                ShaderIo::WriteToColorTarget { slot }.into(),
                &[src_color_xn],
            );
        });
    }
}

/// Checks that the vertex attribute location is unique - doesn't exist yet.
fn ensure_location_is_unique(
    ctx: &Context,
    slot: u32,
    location: Location,
    vertex_buffers: &[RecordedWithIndex<VertexBufferLayoutRecorded>],
) -> Result<(), PipelineError> {
    for vbuf in vertex_buffers {
        for existing_attrib in &vbuf.attribs {
            if location == existing_attrib.location {
                return Err(PipelineError::DuplicateAttribLocation {
                    location: existing_attrib.location,
                    buffer_a: vbuf.slot,
                    buffer_b: slot,
                });
            }
        }
    }
    Ok(())
}

impl Any {
    #[allow(missing_docs)]
    #[track_caller]
    pub fn fill_fragments(&self, location: Location, method: FragmentSampleMethod) -> Any {
        let call_info = call_info!();
        let node = match self.inner() {
            Ok(node) => node,
            Err(reason) => return Any::new_invalid(reason),
        };
        Context::try_with(call_info, |ctx| {
            // disallow fill inside conditional blocks for now. Even if we allow
            // this, it will still cause a stage solver error because the conditional
            // block would have to be in both the vertex (incoming arg) and the
            // fragment (outcoming result) stage, which a single block cannot have.
            // We favor a specific error here, since at the time of writing the
            // stage solver error messages aren't too descriptive yet.
            ctx.push_error_if_outside_encoding_scope("fragment fill");
            let nodes = ctx.pool();
            let ty = &nodes[node].ty;
            let mut render_pipeline = ctx.render_pipeline_mut();

            let vec_ty = match ty {
                Type::Store(StoreType::Sized(SizedType::Vector(len, stype))) => Ok((*len, *stype)),
                _ => Err(PipelineError::InvalidAttributeType(ty.clone())),
            };

            let already_exists = match render_pipeline.find_interpolator(location) {
                Ok(_) => Err(PipelineError::DuplicateInterpolatorLocation(location)),
                Err(_) => Ok(()),
            };

            match (already_exists, vec_ty) {
                (Ok(()), Ok(vec_ty)) => {
                    render_pipeline.interpolators.push((
                        Interpolator {
                            vec_ty,
                            method,
                            location,
                        },
                        call_info,
                    ));
                    drop(nodes); // used mutably in record_node
                    drop(render_pipeline); // used mutably in record_node
                    record_node(call_info, ShaderIo::Interpolate(location).into(), &[*self]);
                    record_node(call_info, ShaderIo::GetInterpolated(location).into(), &[])
                }
                (Err(e), _) | (_, Err(e)) => {
                    ctx.push_error(e.into());
                    Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                }
            }
        })
        .unwrap_or_else(|| Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// see https://www.w3.org/TR/WGSL/#interpolation
pub enum FragmentSamplePosition {
    /// Interpolation is performed at the center of the pixel.
    PixelCenter,
    /// Interpolation is performed at a point that lies at the center of all the usual sample points which are covered by the fragment within the current primitive. This value is the same for all samples in the primitive.
    Centroid,
    /// Interpolation is performed per (multisampling) sample. The generated fragment shader is invoked once per sample when this attribute is applied, which will have a considerable nagative performance impact.
    PerSample,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentSampleMethod {
    Interpolated(Fill, FragmentSamplePosition),
    Flat(PickVertex),
}
