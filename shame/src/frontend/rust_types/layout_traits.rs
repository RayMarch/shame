use crate::any::layout::{Layoutable, LayoutableSized, Repr};
use crate::call_info;
use crate::common::po2::U32PowerOf2;
use crate::common::proc_macro_utils::{self, repr_c_struct_layout, ReprCError, ReprCField};
use crate::frontend::any::render_io::{
    Attrib, VertexBufferLookupIndex, Location, VertexAttribFormat, VertexBufferLayout, VertexLayoutError,
};
use crate::frontend::any::{Any, InvalidReason};
use crate::frontend::encoding::buffer::{BufferAddressSpace, BufferInner, BufferRefInner};
use crate::frontend::encoding::{EncodingError, EncodingErrorKind};
use crate::frontend::error::InternalError;
use crate::frontend::rust_types::len::*;
use crate::ir::ir_type::{
    align_of_array, align_of_array_from_element_alignment, byte_size_of_array_from_stride_len, round_up,
    stride_of_array_from_element_align_size, CanonName, LayoutError, ScalarTypeFp, ScalarTypeInteger,
};
use crate::ir::pipeline::StageMask;
use crate::ir::recording::Context;

use super::array::{Array, Size};
use super::error::FrontendError;
use super::mem::AddressSpace;
use super::reference::{AccessMode, AccessModeReadable};
use super::struct_::{BufferFields, SizedFields, Struct};
use super::type_layout::repr::{TypeRepr, TypeReprStorageOrPacked};
use super::type_layout::layoutable::{self, array_stride, Vector};
use super::type_layout::{
    self, repr, ElementLayout, FieldLayout, FieldLayoutWithOffset, GpuTypeLayout, StructLayout, TypeLayout,
    TypeLayoutSemantics,
};
use super::type_traits::{
    BindingArgs, GpuAligned, GpuSized, GpuStore, GpuStoreImplCategory, NoAtomics, NoBools, NoHandles, VertexAttribute,
};
use super::{len::Len, scalar_type::ScalarType, vec::vec};
use super::{AsAny, GpuType, ToGpuType};
use crate::frontend::any::{shared_io::BindPath, shared_io::BindingType};
use crate::frontend::rust_types::reference::Ref;
use crate::ir::{self, AlignedType, ScalarType as ST, SizedStruct, SizedType, StoreType};
use std::borrow::{Borrow, Cow};
use std::iter::Empty;
use std::mem::size_of;
use std::ops::Deref;
use std::rc::Rc;

/// Types that have a defined memory layout on the Gpu.
///
/// This includes all [`GpuStore`] types, as well as [`PackedVec`] and
/// user defined types which use `#[derive(shame::GpuLayout)]`
///
/// Types that do **not** have a [`GpuLayout`] are pointer or handle types such as
/// [`Ref`], [`Sampler`], [`Texture`] and `[StorageTexture]`
/// and their texture-array counterparts
///
/// # derive macro
/// A `#[derive(shame::GpuLayout)]` macro exists that implements [`GpuLayout`]
/// (among other traits) for user defined structs if all fields of the struct
/// themselves implement [`GpuLayout`].
///
/// ## Example
/// ```
/// use shame as sm;
///
/// #[derive(sm::GpuLayout)]
/// struct PointLight {
///     position: sm::f32x4,
///     intensity: sm::f32x1,
/// }
/// ```
///
/// The derived memory layout follows to the WGSL struct member layout rules
/// described at
/// https://www.w3.org/TR/WGSL/#structure-member-layout
///
/// ## custom alignment and size of fields
///
/// `align` and `size` attributes can be used in front of a struct field
/// to define a minimum alignment and byte-size requirement for that field.
/// ```
/// #[derive(sm::GpuLayout)]
/// struct PointLight {
///     #[align(16)] position: sm::f32x4,
///     #[size(16)] intensity: sm::f32x1,
/// }
///
/// #[derive(sm::GpuLayout)]
/// struct PointLight2 {
///     #[align(2)] // no effect, `position` already has a 16-byte alignment which makes it 2-byte aligned as well
///     position: sm::f32x4,
///     #[size(2)] // no effect, `intensity` is already larger, 4 bytes in size
///     intensity: sm::f32x1,
/// }
/// ```
///
/// ## automatic Layout validation between Cpu and Gpu types
///
/// the `#[cpu(...)]` macro can be used to associate a Cpu type with a Gpu type
/// at the struct declaration level.
/// The equivalence of the two type's [`TypeLayout`]s is validated at pipeline
/// encoding time, as soon as the Gpu types is used in bindings, push-constants or
/// vertex buffers.
///
/// ```
/// #[derive(sm::CpuLayout)]
/// struct PointLightCpu {
///     angle: f32,
///     intensity: f32,
/// }
///
/// #[derive(sm::GpuLayout)]
/// #[cpu(PointLightCpu)] // associate PointLightGpu with PointLightCpu
/// struct PointLightGpu {
///     angle: sm::f32x1,
///     intensity: sm::f32x1,
/// }
/// ```
///
/// # Layout comparison of different types
///
/// The layouts of different [`GpuLayout`]/[`CpuLayout`] types can be compared
/// by comparing [`TypeLayout`] objects returned by `.gpu_layout()`/`.cpu_layout()`
/// ```
/// use shame as sm;
/// use sm::{ GpuLayout, CpuLayout };
///
/// type OnGpu = sm::Array<sm::f32x1, sm::Size<16>>;
/// type OnCpu = [f32; 16];
///
/// if OnGpu::gpu_layout() == OnCpu::cpu_layout() {
///     println!("same layout")
/// }
/// println!("OnGpu:\n{}\n", OnGpu::gpu_layout());
/// println!("OnCpu:\n{}\n", OnCpu::cpu_layout());
/// ```
///
/// [`PackedVec`]: crate::packed::PackedVec
/// [`Ref`]: crate::Ref
/// [`Sampler`]: crate::Sampler
/// [`Texture`]: crate::Texture
/// [`StorageTexture`]: crate::StorageTexture
///
pub trait GpuLayout: Layoutable {
    /// Returns the `Repr` of the `TypeLayout` from `GpuLayout::gpu_layout`.
    ///
    /// `GpuRepr` only exists so a (derived) struct can be packed for use in vertex buffer usage.
    /// It is `repr::Storage` in all other cases.
    type GpuRepr: TypeReprStorageOrPacked;

    /// the `#[cpu(...)]` in `#[derive(GpuLayout)]` allows the definition of a
    /// corresponding Cpu type to the Gpu type that the derive macro is used on.
    ///
    /// If this association exists, this function returns the name and layout of
    /// that Cpu type, otherwise `None` is returned.
    ///
    /// implementor note: if a nested type's `cpu_type_name_and_layout` returns `Some`
    /// this function _MUST NOT_ return `None`, as it would throw away assumptions
    /// the user makes about what this type corresponds to on the cpu.
    /// *** The user expects that layout to be checked! ***
    fn cpu_type_name_and_layout() -> Option<Result<(Cow<'static, str>, TypeLayout), ArrayElementsUnsizedError>>;
}

/// returns a [`TypeLayout`] object that can be used to inspect the layout
/// of a type on the gpu.
///
/// # Layout comparison of different types
///
/// The layouts of different [`GpuLayout`]/[`CpuLayout`] types can be compared
/// by comparing [`TypeLayout`] objects returned by `.gpu_layout()`/`.cpu_layout()`
/// ```
/// use shame as sm;
/// use sm::{ GpuLayout, CpuLayout };
///
/// type OnGpu = sm::Array<sm::f32x1, sm::Size<16>>;
/// type OnCpu = [f32; 16];
///
/// if OnGpu::gpu_layout() == OnCpu::cpu_layout() {
///     println!("same layout")
/// }
/// println!("OnGpu:\n{}\n", OnGpu::gpu_layout());
/// println!("OnCpu:\n{}\n", OnCpu::cpu_layout());
/// ```
pub fn gpu_layout<T: GpuLayout + ?Sized>() -> TypeLayout { gpu_type_layout::<T>().layout() }

pub fn gpu_type_layout<T: GpuLayout + ?Sized>() -> GpuTypeLayout<T::GpuRepr> {
    GpuTypeLayout::new(T::layoutable_type())
}

/// (no documentation yet)
// `CpuLayout::cpu_layout` exists, but this function exists for consistency with
// the `gpu_layout` function. `GpuLayout::gpu_layout` does not exist, so that implementors
// of `GpuLayout` can't overwrite it.
pub fn cpu_layout<T: CpuLayout + ?Sized>() -> TypeLayout { T::cpu_layout() }

pub(crate) fn cpu_type_name_and_layout<T: GpuLayout>(ctx: &Context) -> Option<(Cow<'static, str>, TypeLayout)> {
    match T::cpu_type_name_and_layout().transpose() {
        Ok(t) => t,
        Err(ArrayElementsUnsizedError { elements }) => {
            ctx.push_error(
                InternalError::new(
                    true,
                    format!("`cpu_type_name_and_layout` array elements are unsized: {elements}"),
                )
                .into(),
            );
            None
        }
    }
}

/// returns the `TypeLayout` of `T` and pushes an error to the provided context if it is incompatible with its associated cpu layout
pub(crate) fn get_layout_compare_with_cpu_push_error<T: GpuLayout>(
    ctx: &Context,
    skip_stride_check: bool,
) -> TypeLayout {
    const ERR_COMMENT: &str = "`GpuLayout` uses WGSL layout rules unless #[gpu_repr(packed)] is used.\nsee https://www.w3.org/TR/WGSL/#structure-member-layout\n`CpuLayout` uses #[repr(C)].\nsee https://doc.rust-lang.org/reference/type-layout.html#r-layout.repr.c.struct";

    let gpu_layout = gpu_layout::<T>();
    if let Some((cpu_name, cpu_layout)) = cpu_type_name_and_layout::<T>(ctx) {
        check_layout_push_error(ctx, &cpu_name, &cpu_layout, &gpu_layout, skip_stride_check, ERR_COMMENT).ok();
    }
    gpu_layout
}

pub(crate) fn check_layout_push_error(
    ctx: &Context,
    cpu_name: &str,
    cpu_layout: &TypeLayout,
    gpu_layout: &TypeLayout,
    skip_stride_check: bool,
    comment_on_mismatch_error: &str,
) -> Result<(), InvalidReason> {
    type_layout::eq::check_eq(("cpu", cpu_layout), ("gpu", gpu_layout))
        .map_err(|e| LayoutError::LayoutMismatch(e, Some(comment_on_mismatch_error.to_string())))
        .and_then(|_| {
            if skip_stride_check {
                Ok(())
            } else {
                // the layout is an element in an array, so the strides need to match too
                match (cpu_layout.byte_size(), gpu_layout.byte_size()) {
                    (None, None) | (None, Some(_)) => Err(LayoutError::UnsizedStride { name: cpu_name.into() }),
                    (Some(_), None) => Err(LayoutError::UnsizedStride {
                        name: gpu_layout.short_name(),
                    }),
                    (Some(cpu_size), Some(gpu_size)) => {
                        let cpu_stride = array_stride(cpu_layout.align(), cpu_size);
                        let gpu_stride = array_stride(gpu_layout.align(), gpu_size);

                        if cpu_stride != gpu_stride {
                            Err(LayoutError::StrideMismatch {
                                cpu_name: cpu_name.into(),
                                cpu_stride,
                                gpu_name: gpu_layout.short_name(),
                                gpu_stride,
                            })
                        } else {
                            Ok(())
                        }
                    }
                }
            }
        })
        .map_err(|layout_err| {
            ctx.push_error(layout_err.into());
            InvalidReason::ErrorThatWasPushed
        })
}

/// (no documentation yet)
pub trait CpuLayout {
    // TODO(release) consider making this function "unsafe", since a wrong implementation of it
    // can lead to memory misalignment on the GPU. Even though this is not
    // technically a case where rust demans "unsafe" it follows the general idea of it.
    /// (no documentation yet)
    fn cpu_layout() -> TypeLayout;

    // TODO(release) remove if we decide to not support this function on the `CpuLayout` trait
    // returns `None` if there is no device type associated with `Self`
    //
    // this function is intended to be implemented for types where you have
    // reason to assume that the layouts in `Self::layout()` and `Self::gpu_type_layout()`
    // might disagree, for example because the layout is generated.
    //
    // If you implement `CpuLayout` for your own vector types etc, there is no
    // reason to implement this, since you are *defining* what those types correspond
    // to on the gpu.
    //
    // implementor note: if a nested type returns `Some` this function
    // MUST NOT return `None` as it would throw away assumptions the user
    // makes about what this type corresponds to on device.
    // *** The user expects that layout to be checked! ***
    // TODO(release) atm there is no shame api that lets the user perform Cpu->Gpu checks manually and no
    // `derive(CpuLayout)` `#[gpu(T)]` attribute that corresponds to
    // `derive(GpuLayout)` `#[cpu(T)]`, so consider removing this function entirely
    // fn gpu_type_layout() -> Option<Result<TypeLayout, ArrayElementsUnsizedError>> {
    //     None // no corresponding type to check
    // }
}

#[allow(missing_docs)]
#[derive(Debug)]
pub struct ArrayElementsUnsizedError {
    /// the element type's type layout, which is unsized
    pub elements: TypeLayout,
}

/// (no documentation yet)
pub trait FromAnys {
    /// the required amount of `Any` objects that have to be passed into `Self::from_anys` via iterator, so it won't fail
    fn expected_num_anys() -> usize;

    // construct a `Self` from an iterator of `Any` items.
    //
    // used for instantiating `#[derive(GpuLayout)]` types and [`GpuType`] types
    //
    // compound types such as `Struct<T>` and `Array<T>` expect a **single** `Any` item
    // representing that struct or array, **not** a series of `Any` instances for
    // their contained elements/fields.
    //
    /// (no documentation yet)
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self;
}

/// (no documentation yet)
pub trait GetAllFields {
    // get the individual fields of `Self` or `Ref<Self>` in its `Any` form,
    // i.e. the fields of a `Struct<T>`, the xyzw components of a `vec` etc.
    //
    // returns an empty array if `Self` has no fields.
    //
    // The implementation of this function is not required to do type checking
    // on the incoming or outgoing `Any`s. That part is up to the caller.
    /// (no documentation yet)
    fn fields_as_anys_unchecked(self_as_any: Any) -> impl Borrow<[Any]>;
}

#[track_caller]
pub(crate) fn from_single_any(mut anys: impl Iterator<Item = Any>) -> Any {
    let call_info = call_info!();
    let push_wrong_amount_of_args_error = |amount, call_info| {
        Context::try_with(call_info, |ctx| {
            ctx.push_error_get_invalid_any(FrontendError::InvalidDowncastAmount { amount }.into())
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    };

    match anys.next() {
        None => push_wrong_amount_of_args_error(0, call_info),
        Some(any) => match anys.next() {
            None => any,
            Some(any) => {
                let count = anys.count() + 2; // we called `next()` successfully twice
                push_wrong_amount_of_args_error(count, call_info)
            }
        },
    }
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid vertex layout. Valid layouts are non-bool `shame::vec`, `shame::PackedVec` \
    or any combination of these in a `#[derive(GpuLayout)]` struct."
)]
//TODO(release) seal this trait
/// This trait is implemented by:
///
/// * `sm::vec`s of non-boolean type (e.g. `sm::f32x4`)
/// * `sm::packed::PackedVec`s (e.g. `sm::packed::unorm8x4`)
/// * `#[derive(sm::GpuLayout)]` structs that contains only elements of the above mentioned types
pub trait VertexLayout: GpuLayout + FromAnys {}
impl<T: VertexAttribute> VertexLayout for T {}

// #[derive(GpuLayout)]
// TODO(release) remove this type. This is an example impl for figuring out how the derive macro should work
#[derive(Clone)]
pub struct GpuT {
    a: vec<f32, x1>,
    b: vec<u32, x1>,
    c: Array<vec<i32, x1>, Size<4>>,
}

// TODO(release) remove this type. This is an example impl for figuring out how the derive macro should work
#[derive(Clone, Copy)]
pub struct GpuTypeRef<AS: AddressSpace, AM: AccessMode> {
    a: Ref<vec<f32, x1>, AS, AM>,
    b: Ref<vec<u32, x1>, AS, AM>,
    c: Ref<Array<vec<i32, x1>, Size<4>>, AS, AM>,
}

impl VertexLayout for GpuT
where
    vec<f32, x1>: for<'trivial_bound> VertexAttribute,
    vec<u32, x1>: for<'trivial_bound> VertexAttribute,
    Array<vec<i32, x1>, Size<4>>: for<'trivial_bound> VertexAttribute,
{
}

impl<AS: AddressSpace, AM: AccessMode> FromAnys for GpuTypeRef<AS, AM> {
    fn expected_num_anys() -> usize { 3 }

    fn from_anys(anys: impl Iterator<Item = Any>) -> Self {
        use crate::__private::proc_macro_reexports::{collect_into_array_exact, push_wrong_amount_of_args_error};

        const EXPECTED_LEN: usize = 3;
        assert_eq!(EXPECTED_LEN, Self::expected_num_anys());
        let [a, b, c] = match collect_into_array_exact::<Any, EXPECTED_LEN>(anys) {
            Ok(t) => t,
            Err(actual_len) => {
                push_wrong_amount_of_args_error(actual_len, EXPECTED_LEN, call_info!());
                [Any::new_invalid(InvalidReason::ErrorThatWasPushed); EXPECTED_LEN]
            }
        };

        Self {
            a: From::from(a),
            b: From::from(b),
            c: From::from(c),
        }
    }
}

impl NoHandles for GpuT {}
impl NoAtomics for GpuT {}
impl NoBools for GpuT {}

impl SizedFields for GpuT
where
    vec<f32, x1>: for<'trivial_bound> GpuSized,
    vec<u32, x1>: for<'trivial_bound> GpuSized,
    Array<vec<i32, x1>, Size<4>>: for<'trivial_bound> GpuSized,
{
    fn get_sizedstruct_type() -> ir::SizedStruct {
        let struct_ = ir::SizedStruct::new_nonempty(
            std::stringify!(GpuType).into(),
            vec![
                ir::SizedField::new("a".into(), None, None, <vec<f32, x1> as GpuSized>::sized_ty()),
                ir::SizedField::new("b".into(), None, None, <vec<u32, x1> as GpuSized>::sized_ty()),
            ],
            ir::SizedField::new("c".into(), None, None, <vec<i32, x1> as GpuSized>::sized_ty()),
        );
        match struct_ {
            Ok(s) => s,
            Err(ir::StructureFieldNamesMustBeUnique) => unreachable!("field names are assumed unique"),
        }
    }
}

impl BufferFields for GpuT {
    fn as_anys(&self) -> impl Borrow<[Any]> { [self.a.as_any(), self.b.as_any(), self.c.as_any()] }

    #[allow(clippy::clone_on_copy)]
    fn clone_fields(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
            c: self.c.clone(),
        }
    }

    fn get_bufferblock_type() -> ir::BufferBlock {
        // compiler_error! if the struct has zero fields!

        let a = (
            std::stringify!(a).into(),
            <vec<f32, x1> as GpuSized>::sized_ty(),
            Some(32),
            Some(32).map(|a| a.try_into().expect("power of two validated during codegen")),
        );
        let b = (
            std::stringify!(b).into(),
            <vec<u32, x1> as GpuSized>::sized_ty(),
            None,
            None,
        );

        // generate the following line if a custom-size was provided:
        // this will trigger a compipler error if the type is not `shame::Sized`
        let c = (
            std::stringify!(b).into(),
            <vec<i32, x1> as GpuAligned>::aligned_ty(),
            None,
            None,
        );

        // otherwise generate this line:
        // let c = (std::stringify!(b).into(), <vec<i32, x1> as crate::Aligned>::aligned_ty(), None, None,);

        let mut fields = vec![
            {
                let (name, ty, custom_min_size, custom_min_align) = a;
                crate::ir::SizedField {
                    name,
                    custom_min_size,
                    custom_min_align,
                    ty,
                }
            },
            {
                let (name, ty, custom_min_size, custom_min_align) = b;
                crate::ir::SizedField {
                    name,
                    custom_min_size,
                    custom_min_align,
                    ty,
                }
            },
        ];

        let mut last_unsized = None;
        #[allow(clippy::no_effect)]
        {
            Some(4);
            <vec<i32, x1> as GpuSized>::sized_ty;
            let (name, ty, custom_min_size, custom_min_align) = c;
            match ty {
                crate::ir::AlignedType::Sized(ty) => fields.push(crate::ir::SizedField {
                    name,
                    custom_min_size,
                    custom_min_align,
                    ty,
                }),
                crate::ir::AlignedType::RuntimeSizedArray(element_ty) => {
                    last_unsized = Some(crate::ir::RuntimeSizedArrayField {
                        name,
                        custom_min_align,
                        element_ty,
                    })
                }
            }
        }

        use ir::ir_type::BufferBlockDefinitionError as E;
        match ir::BufferBlock::new(std::stringify!(GpuType).into(), fields, last_unsized) {
            Ok(t) => t,
            Err(e) => match e {
                E::MustHaveAtLeastOneField => unreachable!(">= 1 field is ensured by derive macro"),
                E::FieldNamesMustBeUnique => unreachable!("unique field idents are ensured by rust struct definition"),
            },
        }
    }
}

impl GetAllFields for GpuT
where
    vec<f32, x1>: for<'trivial_bound> GpuType,
    vec<u32, x1>: for<'trivial_bound> GpuType,
    Array<vec<i32, x1>, Size<4>>: for<'trivial_bound> GpuType,
{
    fn fields_as_anys_unchecked(self_: Any) -> impl Borrow<[Any]> {
        [
            self_.get_field("a".into()),
            self_.get_field("b".into()),
            self_.get_field("c".into()),
        ]
    }
}

impl LayoutableSized for GpuT {
    fn layoutable_type_sized() -> layoutable::SizedType { todo!() }
}
impl Layoutable for GpuT {
    fn layoutable_type() -> layoutable::LayoutableType { todo!() }
}

impl GpuLayout for GpuT {
    // fn gpu_repr() -> Repr { todo!() }
    type GpuRepr = repr::Storage;

    fn cpu_type_name_and_layout() -> Option<Result<(Cow<'static, str>, TypeLayout), ArrayElementsUnsizedError>> {
        Some(Ok((
            std::stringify!(RustType).into(),
            <RustType as CpuLayout>::cpu_layout(),
        )))
    }
}

impl FromAnys for GpuT {
    fn expected_num_anys() -> usize { 3 }

    #[track_caller]
    fn from_anys(mut anys: impl Iterator<Item = Any>) -> Self {
        use crate::__private::proc_macro_reexports::{collect_into_array_exact, push_wrong_amount_of_args_error};

        const EXPECTED_LEN: usize = 3;
        assert_eq!(EXPECTED_LEN, Self::expected_num_anys());
        let [a, b, c] = match collect_into_array_exact::<Any, EXPECTED_LEN>(anys) {
            Ok(t) => t,
            Err(actual_len) => {
                push_wrong_amount_of_args_error(actual_len, EXPECTED_LEN, call_info!());
                [Any::new_invalid(InvalidReason::ErrorThatWasPushed); EXPECTED_LEN]
            }
        };

        Self {
            a: From::from(a),
            b: From::from(b),
            c: From::from(c),
        }
    }
}

impl<T: SizedFields + NoAtomics> ToGpuType for T {
    type Gpu = Struct<T>;

    #[track_caller]
    fn to_gpu(&self) -> Self::Gpu {
        Struct::<T>::from(Any::new_struct(T::get_sizedstruct_type(), self.as_anys().borrow()))
    }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

impl GpuStore for GpuT {
    type RefFields<AS: AddressSpace, AM: AccessMode> = GpuTypeRef<AS, AM>;

    fn store_ty() -> ir::StoreType
    where
        Self: for<'triv> GpuType,
    {
        unreachable!()
    }

    fn instantiate_buffer_inner<AS: BufferAddressSpace>(
        args: Result<BindingArgs, InvalidReason>,
        bind_ty: BindingType,
    ) -> BufferInner<Self, AS>
    where
        Self: for<'trivial_bound> NoAtomics + for<'trivial_bound> NoBools,
    {
        BufferInner::new_fields(args, bind_ty)
    }

    fn instantiate_buffer_ref_inner<AS: BufferAddressSpace, AM: AccessModeReadable>(
        args: Result<BindingArgs, InvalidReason>,
        bind_ty: BindingType,
    ) -> BufferRefInner<Self, AS, AM>
    where
        Self: for<'trivial_bound> NoBools,
    {
        BufferRefInner::new_fields(args, bind_ty)
    }

    fn impl_category() -> GpuStoreImplCategory { GpuStoreImplCategory::Fields(Self::get_bufferblock_type()) }
}

impl GpuAligned for GpuT {
    fn aligned_ty() -> ir::AlignedType
    where
        Self: for<'trivial_bound> GpuType,
    {
        unreachable!("Self: !GpuType")
    }
}

impl GpuSized for GpuT {
    fn sized_ty() -> ir::SizedType
    where
        Self: for<'trivial_bound> GpuType,
    {
        unreachable!("Self: !GpuType")
    }
}

// #[derive(HostLayout)]
// TODO(release) remove. This is an example impl
pub struct RustType {
    a: f32,
    b: i32,
}


impl CpuLayout for RustType
where
    f32: CpuAligned,
    i32: CpuAligned,
{
    fn cpu_layout() -> TypeLayout {
        use CpuAligned;
        let layout = repr_c_struct_layout(
            None,
            std::stringify!(RustType),
            &[(
                ReprCField {
                    name: std::stringify!(a),
                    alignment: f32::CPU_ALIGNMENT,
                    layout: f32::cpu_layout(),
                },
                std::mem::offset_of!(RustType, a),
                std::mem::size_of::<f32>(),
            )],
            ReprCField {
                name: std::stringify!(b),
                alignment: i32::CPU_ALIGNMENT,
                layout: i32::cpu_layout(),
            },
            i32::CPU_SIZE,
        );

        match layout {
            Ok(l) => l,
            Err(e) => match e {
                ReprCError::SecondLastElementIsUnsized => {
                    unreachable!("`offset_of` was called on this elmement, so it must be sized")
                }
            },
        }
    }

    //fn gpu_type_layout() -> Option<Result<TypeLayout, ArrayElementsUnsizedError>> { Some(Ok(GpuT::gpu_layout())) }
}

impl CpuLayout for f32 {
    fn cpu_layout() -> TypeLayout {
        TypeLayout::from_rust_sized::<f32>(TypeLayoutSemantics::Vector(Vector::new(
            layoutable::ScalarType::F32,
            layoutable::Len::X1,
        )))
    }
    // fn gpu_type_layout() -> Option<Result<TypeLayout, ArrayElementsUnsizedError>> {
    //     Some(Ok(vec::<Self, x1>::gpu_layout()))
    // }
}

impl CpuLayout for f64 {
    fn cpu_layout() -> TypeLayout {
        TypeLayout::from_rust_sized::<f32>(TypeLayoutSemantics::Vector(Vector::new(
            layoutable::ScalarType::F64,
            layoutable::Len::X1,
        )))
    }
}

impl CpuLayout for u32 {
    fn cpu_layout() -> TypeLayout {
        TypeLayout::from_rust_sized::<f32>(TypeLayoutSemantics::Vector(Vector::new(
            layoutable::ScalarType::U32,
            layoutable::Len::X1,
        )))
    }
}

impl CpuLayout for i32 {
    fn cpu_layout() -> TypeLayout {
        TypeLayout::from_rust_sized::<f32>(TypeLayoutSemantics::Vector(Vector::new(
            layoutable::ScalarType::I32,
            layoutable::Len::X1,
        )))
    }
}

/// (no documentation yet)
pub trait CpuAligned {
    /// (no documentation yet)
    const CPU_ALIGNMENT: U32PowerOf2;
    /// (no documentation yet)
    const CPU_SIZE: Option<usize>;
    /// (no documentation yet)
    fn alignment() -> U32PowerOf2;
}

impl<T> CpuAligned for T {
    const CPU_ALIGNMENT: U32PowerOf2 =
        U32PowerOf2::try_from_usize(std::mem::align_of::<T>()).expect("alignment of types is always a power of 2");
    const CPU_SIZE: Option<usize> = Some(std::mem::size_of::<T>());
    fn alignment() -> U32PowerOf2 { Self::CPU_ALIGNMENT }
}

impl<T> CpuAligned for [T] {
    // must be same as align of `T` since `std::slice::from_ref` and `&slice[0]` exist
    const CPU_ALIGNMENT: U32PowerOf2 = T::CPU_ALIGNMENT;
    const CPU_SIZE: Option<usize> = None;
    fn alignment() -> U32PowerOf2 {
        U32PowerOf2::try_from_usize(std::mem::align_of_val::<[T]>(&[]))
            .expect("alignment of types is always a power of 2")
    }
}

impl<T: CpuLayout + Sized, const N: usize> CpuLayout for [T; N] {
    fn cpu_layout() -> TypeLayout {
        let align = U32PowerOf2::try_from(<Self as CpuAligned>::alignment() as u32).unwrap();

        TypeLayout::new(
            Some(std::mem::size_of::<Self>() as u64),
            align,
            TypeLayoutSemantics::Array(
                Rc::new(ElementLayout {
                    byte_stride: std::mem::size_of::<T>() as u64,
                    ty: T::cpu_layout(),
                }),
                Some(u32::try_from(N).expect("arrays larger than u32::MAX elements are not supported by WGSL")),
            ),
        )
    }

    // fn gpu_type_layout() -> Option<Result<TypeLayout, ArrayElementsUnsizedError>> {
    //     let t_on_dev = match T::gpu_type_layout()? {
    //         Ok(t) => t,
    //         Err(e) => return Some(Err(e)),
    //     };

    //     let Some(t_size) = t_on_dev.byte_size() else {
    //         return Some(Err(ArrayElementsUnsizedError { elements: t_on_dev }));
    //     };
    //     let stride = stride_of_array_from_element_align_size(t_on_dev.align(), N as u64);
    //     let len: u64 = N as u64;
    //     let layout = TypeLayout::new(
    //         Some(byte_size_of_array_from_stride_len(stride, len)),
    //         align_of_array_from_element_alignment(t_on_dev.align()),
    //         TypeLayoutSemantics::Array(
    //             Rc::new(ElementLayout {
    //                 byte_stride: stride_of_array_from_element_align_size(t_on_dev.align(), t_size),
    //                 ty: t_on_dev,
    //             }),
    //             Some(u32::try_from(N).expect("arrays larger than u32::MAX elements are not supported by WGSL")),
    //         ),
    //     );
    //     Some(Ok(layout))
    // }
}

impl<T: CpuLayout + Sized> CpuLayout for [T] {
    fn cpu_layout() -> TypeLayout {
        let align = U32PowerOf2::try_from(<Self as CpuAligned>::alignment() as u32).unwrap();

        TypeLayout::new(
            None,
            align,
            TypeLayoutSemantics::Array(
                Rc::new(ElementLayout {
                    byte_stride: std::mem::size_of::<T>() as u64,
                    ty: T::cpu_layout(),
                }),
                None,
            ),
        )
    }

    // TODO(release) remove if we decide to not support this function on the `CpuLayout` trait
    // fn gpu_type_layout() -> Option<Result<TypeLayout, ArrayElementsUnsizedError>> {
    //     let t_on_dev = match T::gpu_type_layout()? {
    //         Ok(t) => t,
    //         Err(e) => return Some(Err(e)),
    //     };

    //     let Some(t_size) = t_on_dev.byte_size() else {
    //         return Some(Err(ArrayElementsUnsizedError { elements: t_on_dev }));
    //     };
    //     let layout = TypeLayout::new(
    //         None,
    //         align_of_array_from_element_alignment(t_on_dev.align()),
    //         TypeLayoutSemantics::Array(
    //             Rc::new(ElementLayout {
    //                 byte_stride: stride_of_array_from_element_align_size(t_on_dev.align(), t_size),
    //                 ty: t_on_dev,
    //             }),
    //             None,
    //         ),
    //     );
    //     Some(Ok(layout))
    // }
}
