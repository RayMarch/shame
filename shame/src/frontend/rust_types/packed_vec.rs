#![allow(non_camel_case_types)]
use std::{
    borrow::Cow,
    marker::{PhantomData, PhantomPinned},
};

use crate::{
    any::{
        layout::{Layoutable, LayoutableSized},
        AsAny, DataPackingFn,
    },
    common::floating_point::f16,
    f32x2, f32x4, gpu_layout, i32x4, u32x1, u32x4,
};
use crate::frontend::rust_types::len::{x1, x2, x3, x4};
use crate::frontend::rust_types::vec::vec;
use crate::{
    call_info,
    frontend::any::{render_io::VertexAttribFormat, Any, InvalidReason},
    ir::{self, recording::Context, PackedVector, SizedType},
};

use super::{
    error::FrontendError,
    layout_traits::{from_single_any, ArrayElementsUnsizedError, FromAnys, GpuLayout},
    len::LenEven,
    scalar_type::ScalarType,
    type_layout::{self, layoutable, repr, Repr, TypeLayout, TypeLayoutSemantics},
    type_traits::{GpuAligned, GpuSized, NoAtomics, NoBools, NoHandles, VertexAttribute},
    vec::IsVec,
    GpuType,
};

/// stores a `vec` using less memory.
/// `PackedVec` types can be used as `VertexAttribute`s in vertex buffers.
/// use the `.unpack()` associated function to unpack the full `vec` value
///
/// choose `L` as either
/// - `shame::x2`,
/// - `shame::x4`,
///
/// choose `T` as either
/// - `u8`
/// - `u16`
/// - `i8`
/// - `i16`
/// - `shame::unorm8`
/// - `shame::unorm16`
#[derive(Clone, Copy)]
pub struct PackedVec<T: PackedScalarType, L: LenEven>(Unpackable<T, L>);

/// either unpacked, or packed + unpack-function
#[derive(Clone, Copy)]
pub(crate) enum Unpackable<T: PackedScalarType, L: LenEven> {
    Unpacked(vec<T::Unpacked, L>),
    PackedU32(vec<u32, x1>),
}

/// implemented for all `PackedVec` types that can be represented as a `Value` in
/// the shader in their packed state, and be unpacked via a shader function.
/// (see https://www.w3.org/TR/WGSL/#unpack-builtin-functions)
pub trait UnpackableValue {
    type PackedShaderRepr: IsVec;
    type Unpacked: IsVec;

    fn unpack_builtin_function(packed: Self::PackedShaderRepr) -> Self::Unpacked;
}

impl<T: PackedScalarType, L: LenEven> PackedVec<T, L> {
    /// the value inside of `self` converted to a regular [`vec`]
    pub fn unpack(&self) -> vec<T::Unpacked, L> {
        match self.0 {
            Unpackable::Unpacked(val) => val,
            Unpackable::PackedU32(val) => Context::try_with(call_info!(), |ctx| {
                // this can be implemented for all the types that have a
                // corresponding "unpack*" wgsl builtin function, also
                // a corresponding "pack" function needs to be defined. Once this is
                // implemented, the PackedU32 variant can be used to make
                // PackedVec fully compatible with storage and uniform buffers.
                ctx.push_error_get_invalid_any(FrontendError::PackedVecU32UnpackingNotSupportedYet.into())
            })
            .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
            .into(),
        }
    }
}

impl<T: PackedScalarType, L: LenEven> PackedVec<T, L> {
    /// returns an ir::SizedType with identical size and alignment
    ///
    /// `packed_vec` is a type that does not exist in the shader type system.
    /// As such it will never show up inside a shader, but for layout calculations
    /// it still provides a type that has the same size and alignment characteristics.
    ///
    /// > implementors note:
    /// > a proper solution to this would be to have 2 distinct runtime type systems,
    /// > one for everything including binding-types, and one for just within the shader
    /// > with a lossy conversion in between. For now there is no time for that.
    /// > note: this now exists, it is shame::TypeLayout
    fn sized_ty_equivalent() -> SizedType {
        use ir::ir_type::PackedVectorByteSize as Size;
        match get_type_description::<L, T>().byte_size() {
            Size::_2 => <vec<f16, x1> as GpuSized>::sized_ty(),
            Size::_4 => <vec<u32, x1> as GpuSized>::sized_ty(),
            Size::_8 => <vec<u32, x2> as GpuSized>::sized_ty(),
        }
    }
}

pub(crate) fn get_type_description<L: LenEven, T: PackedScalarType>() -> PackedVector {
    PackedVector {
        len: L::LEN_EVEN,
        bits_per_component: T::BITS_PER_COMPONENT,
        scalar_type: T::SCALAR_TYPE,
    }
}

impl<T: PackedScalarType, L: LenEven> GpuSized for PackedVec<T, L> {
    fn sized_ty() -> ir::SizedType
    where
        Self: GpuType,
    {
        unreachable!("Self: !GpuType")
    }
}

impl<T: PackedScalarType, L: LenEven> GpuAligned for PackedVec<T, L> {
    // TODO(release) shouldn't this add a Self: GpuType bound as well, like GpuSized does?
    //     that way we can maybe get rid of the sized_ty_equivalent workaround
    fn aligned_ty() -> ir::AlignedType { ir::AlignedType::Sized(Self::sized_ty_equivalent()) }
}

impl<T: PackedScalarType, L: LenEven> NoBools for PackedVec<T, L> {}
impl<T: PackedScalarType, L: LenEven> NoHandles for PackedVec<T, L> {}
impl<T: PackedScalarType, L: LenEven> NoAtomics for PackedVec<T, L> {}
impl<T: PackedScalarType, L: LenEven> LayoutableSized for PackedVec<T, L> {
    fn layoutable_type_sized() -> layoutable::SizedType {
        layoutable::PackedVector {
            scalar_type: T::SCALAR_TYPE,
            bits_per_component: T::BITS_PER_COMPONENT,
            len: L::LEN_EVEN,
        }
        .into()
    }
}
impl<T: PackedScalarType, L: LenEven> Layoutable for PackedVec<T, L> {
    fn layoutable_type() -> layoutable::LayoutableType { Self::layoutable_type_sized().into() }
}

impl<T: PackedScalarType, L: LenEven> GpuLayout for PackedVec<T, L> {
    type GpuRepr = repr::Storage;

    fn cpu_type_name_and_layout() -> Option<Result<(Cow<'static, str>, TypeLayout), ArrayElementsUnsizedError>> {
        let sized_ty: layoutable::SizedType = Self::layoutable_type_sized();
        let name = sized_ty.to_string().into();
        let layout = TypeLayout::new_layout_for(&sized_ty.into(), Repr::Storage);
        Some(Ok((name, layout)))
    }
}

impl<T: PackedScalarType, L: LenEven> FromAnys for PackedVec<T, L> {
    fn expected_num_anys() -> usize { 1 }

    #[track_caller]
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self { from_single_any(anys).into() }
}

impl<T: PackedScalarType, L: LenEven> From<Any> for PackedVec<T, L> {
    #[track_caller]
    fn from(any: Any) -> Self {
        let inner = Context::try_with(call_info!(), |ctx| {
            let err = |ty| {
                ctx.push_error_get_invalid_any(
                    FrontendError::InvalidDowncastToNonShaderType(ty, gpu_layout::<Self>()).into(),
                )
            };
            match any.ty() {
                None => Unpackable::Unpacked(any.into()),
                Some(ty) => match ty {
                    ir::Type::Store(ir::StoreType::Sized(ir::SizedType::Vector(len, t))) => {
                        // if we get an u32x1 (with future wgpu vertex formats supported, like uint8x1) then
                        // thats an unpacked type, not a packed type that uses u32 as a neutral "bytes" type.
                        let t_unpacked = T::Unpacked::SCALAR_TYPE;

                        let is_u32x1_unpacked = t_unpacked == u32::SCALAR_TYPE && L::LEN == ir::Len::X1;
                        match (len, t) {
                            (ir::Len::X1, ir::ScalarType::U32) if is_u32x1_unpacked => match is_u32x1_unpacked {
                                true => Unpackable::Unpacked(any.into()),
                                false => Unpackable::PackedU32(any.into()),
                            },
                            _ if len == L::LEN && t == t_unpacked => Unpackable::Unpacked(any.into()),
                            _ => Unpackable::Unpacked(err(ty).into()),
                        }
                    }
                    _ => Unpackable::Unpacked(err(ty).into()),
                },
            }
        })
        .unwrap_or(Unpackable::Unpacked(
            Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding).into(),
        ));

        Self(inner)
    }
}

#[diagnostic::on_unimplemented(message = "scalar type `{Self}` cannot be used to pack a vector")]
pub trait PackedScalarType: Copy {
    type Unpacked: ScalarType;
    const SCALAR_TYPE: ir::PackedScalarType;
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent;
}

impl PackedScalarType for i8 {
    type Unpacked = i32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Int;
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_8;
}
impl PackedScalarType for i16 {
    type Unpacked = i32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Int;
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_16;
}
impl PackedScalarType for u8 {
    type Unpacked = u32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Uint;
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_8;
}
impl PackedScalarType for u16 {
    type Unpacked = u32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Uint;
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_16;
}

#[derive(Clone, Copy)]
/// A marker type for components in a [`PackedVec`]. Only valid to use within [`PackedVec`].
///
/// Example: [`PackedVec<unorm8, x2>`] (or use the [`unorm8x2`] alias).
///
/// It represents `u8` components whose `0..=255` range gets converted to
/// float `0.0..=1.0` `f32` when unpacked on the gpu.
pub struct unorm8;
impl PackedScalarType for unorm8 {
    type Unpacked = f32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Float(ir::PackedFloat::Unorm);
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_8;
}
#[derive(Clone, Copy)]
/// A marker type for components in a [`PackedVec`]. Only valid to use within [`PackedVec`].
///
/// Example: [`PackedVec<snorm8, x2>`] (or use the [`snorm8x2`] alias).
///
/// It represents `i8` components whose `-127..=127` range gets converted to
/// float `0.0..=1.0` `f32` when unpacked on the gpu.
pub struct snorm8;
impl PackedScalarType for snorm8 {
    type Unpacked = f32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Float(ir::PackedFloat::Snorm);
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_8;
}
#[derive(Clone, Copy)]
/// A marker type for components in a [`PackedVec`]. Only valid to use within [`PackedVec`].
///
/// Example: [`PackedVec<unorm16, x2>`] (or use the [`unorm16x2`] alias).
///
/// It represents `u16` components whose `0..=65535` range gets converted to
/// float `0.0..=1.0` `f32` when unpacked on the gpu.
/// (no documentation yet)
pub struct unorm16;
impl PackedScalarType for unorm16 {
    type Unpacked = f32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Float(ir::PackedFloat::Unorm);
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_16;
}
#[derive(Clone, Copy)]
/// A marker type for components in a [`PackedVec`]. Only valid to use within [`PackedVec`].
///
/// Example: [`PackedVec<snorm16, x2>`] (or use the [`snorm16x2`] alias).
///
/// It represents `i16` components whose `-32767..=32767` range gets converted to
/// float `0.0..=1.0` `f32` when unpacked on the gpu.
/// (no documentation yet)
pub struct snorm16;
impl PackedScalarType for snorm16 {
    type Unpacked = f32;
    const SCALAR_TYPE: ir::PackedScalarType = ir::PackedScalarType::Float(ir::PackedFloat::Snorm);
    const BITS_PER_COMPONENT: ir::PackedBitsPerComponent = ir::PackedBitsPerComponent::_16;
}

impl<T: PackedScalarType, L: LenEven> VertexAttribute for PackedVec<T, L>
where
    Self: NoBools,
{
    fn vertex_attrib_format() -> VertexAttribFormat { VertexAttribFormat::Coarse(get_type_description::<L, T>()) }
}

/// see https://www.w3.org/TR/WGSL/#pack4x8snorm-builtin
#[track_caller]
pub fn u32_bits_of_snorm8x4_from_f32x4(e: f32x4) -> u32x1 { e.as_any().pack_data(DataPackingFn::Pack4x8snorm).into() }

/// see https://www.w3.org/TR/WGSL/#pack4x8unorm-builtin
#[track_caller]
pub fn u32_bits_of_unorm8x4_from_f32x4(e: f32x4) -> u32x1 { e.as_any().pack_data(DataPackingFn::Pack4x8unorm).into() }

/// see https://www.w3.org/TR/WGSL/#pack4xI8-builtin
#[track_caller]
pub fn u32_bits_of_i8x4_from_i32x4(e: i32x4) -> u32x1 { e.as_any().pack_data(DataPackingFn::Pack4xI8).into() }

/// see https://www.w3.org/TR/WGSL/#pack4xU8-builtin
#[track_caller]
pub fn u32_bits_of_u8x4_from_u32x4(e: u32x4) -> u32x1 { e.as_any().pack_data(DataPackingFn::Pack4xU8).into() }

/// see https://www.w3.org/TR/WGSL/#pack4xI8Clamp-builtin
#[track_caller]
pub fn u32_bits_of_i8x4_clamp_from_i32x4(e: i32x4) -> u32x1 {
    e.as_any().pack_data(DataPackingFn::Pack4xI8Clamp).into()
}

/// see https://www.w3.org/TR/WGSL/#pack4xU8Clamp-builtin
#[track_caller]
pub fn u32_bits_of_u8x4_clamp_from_u32x4(e: u32x4) -> u32x1 {
    e.as_any().pack_data(DataPackingFn::Pack4xU8Clamp).into()
}

/// see https://www.w3.org/TR/WGSL/#pack2x16snorm-builtin
#[track_caller]
pub fn u32_bits_of_snorm16x2_from_f32x2(e: f32x2) -> u32x1 { e.as_any().pack_data(DataPackingFn::Pack2x16snorm).into() }

/// see https://www.w3.org/TR/WGSL/#pack2x16unorm-builtin
#[track_caller]
pub fn u32_bits_of_unorm16x2_from_f32x2(e: f32x2) -> u32x1 { e.as_any().pack_data(DataPackingFn::Pack2x16unorm).into() }

/// see https://www.w3.org/TR/WGSL/#pack2x16float-builtin
#[track_caller]
pub fn u32_bits_of_f16x2_from_f32x2(e: f32x2) -> u32x1 { e.as_any().pack_data(DataPackingFn::Pack2x16float).into() }

/// see https://www.w3.org/TR/WGSL/#unpack4x8snorm-builtin
#[track_caller]
pub fn u32_bits_of_snorm8x4_to_f32x4(e: u32x1) -> f32x4 { e.as_any().pack_data(DataPackingFn::Unpack4x8snorm).into() }

/// see https://www.w3.org/TR/WGSL/#unpack4x8unorm-builtin
#[track_caller]
pub fn u32_bits_of_unorm8x4_to_f32x4(e: u32x1) -> f32x4 { e.as_any().pack_data(DataPackingFn::Unpack4x8unorm).into() }

/// see https://www.w3.org/TR/WGSL/#unpack4xI8-builtin
#[track_caller]
pub fn u32_bits_of_i8x4_to_i32x4(e: u32x1) -> i32x4 { e.as_any().pack_data(DataPackingFn::Unpack4xI8).into() }

/// see https://www.w3.org/TR/WGSL/#unpack4xU8-builtin
#[track_caller]
pub fn u32_bits_of_u8x4_to_u32x4(e: u32x1) -> u32x4 { e.as_any().pack_data(DataPackingFn::Unpack4xU8).into() }

/// see https://www.w3.org/TR/WGSL/#unpack2x16snorm-builtin
#[track_caller]
pub fn u32_bits_of_snorm16x2_to_f32x2(e: u32x1) -> f32x2 { e.as_any().pack_data(DataPackingFn::Unpack2x16snorm).into() }

/// see https://www.w3.org/TR/WGSL/#unpack2x16unorm-builtin
#[track_caller]
pub fn u32_bits_of_unorm16x2_to_f32x2(e: u32x1) -> f32x2 { e.as_any().pack_data(DataPackingFn::Unpack2x16unorm).into() }

/// see https://www.w3.org/TR/WGSL/#unpack2x16float-builtin
#[track_caller]
pub fn u32_bits_of_f16x2_to_f32x2(e: u32x1) -> f32x2 { e.as_any().pack_data(DataPackingFn::Unpack2x16float).into() }

/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type  unorm8x2 = PackedVec< unorm8, x2>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type  unorm8x4 = PackedVec< unorm8, x4>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type unorm16x2 = PackedVec<unorm16, x2>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type unorm16x4 = PackedVec<unorm16, x4>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type  snorm8x2 = PackedVec< snorm8, x2>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type  snorm8x4 = PackedVec< snorm8, x4>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type snorm16x2 = PackedVec<snorm16, x2>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type snorm16x4 = PackedVec<snorm16, x4>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type      u8x2 = PackedVec<     u8, x2>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type      u8x4 = PackedVec<     u8, x4>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type     u16x2 = PackedVec<    u16, x2>;
/// (no documentation yet, see [`PackedVec`])
#[rustfmt::skip] pub type     u16x4 = PackedVec<    u16, x4>;
