#![allow(non_snake_case, clippy::match_like_matches_macro)]

mod derive_layout;
mod util;

use derive_layout::WhichDerive;
use proc_macro::TokenStream;
use syn::parse_macro_input;

use syn::spanned::*;
use syn::Data;
use syn::Fields;

/// implements [`GpuLayout`] and other traits for user defined structs
/// if all fields of the struct themselves implement [`GpuLayout`].
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
/// ## other traits
///
/// This macro conditionally implements
/// - [`ToGpuType`] where `Self::Gpu = Struct<Self>`
/// - [`GpuSized`]
///   if all fields are [`GpuSized`]
/// - [`GpuAligned`]
///   if all fields are [`GpuAligned`]
/// - [`NoBools`], [`NoHandles`], [`NoAtomics`]
///   if all fields implement those respective traits too
///
/// as well as some other traits that are used internally
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
/// ## Automatic Layout validation between Cpu and Gpu types
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
/// [`PackedVec`]: shame::packed::PackedVec
/// [`Ref`]: shame::Ref
/// [`Sampler`]: shame::Sampler
/// [`Texture`]: shame::Texture
/// [`StorageTexture`]: shame::StorageTexture
/// [`ToGpuType`]: shame::ToGpuType
/// [`GpuSized`]: shame::GpuSized
/// [`GpuAligned`]: shame::GpuAligned
/// [`NoBools`]: shame::NoBools
/// [`NoHandles`]: shame::NoHandles
/// [`NoAtomics`]: shame::NoAtomics
#[proc_macro_derive(GpuLayout, attributes(size, align, cpu, gpu_repr))]
pub fn derive_gpu_layout(input: TokenStream) -> TokenStream { derive_impl(WhichDerive::GpuLayout, input) }

#[proc_macro_derive(CpuLayout, attributes())]
pub fn derive_cpu_layout(input: TokenStream) -> TokenStream { derive_impl(WhichDerive::CpuLayout, input) }

fn derive_impl(which_derive: WhichDerive, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let span = input.span();

    match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(named_fields) => {
                derive_layout::impl_for_struct(which_derive, &input, &span, data_struct, named_fields)
            }
            Fields::Unnamed(_) | Fields::Unit => {
                Err(syn::Error::new(span, "Must be used on a struct with named fields"))
            }
        },
        Data::Union(_) | Data::Enum(_) => Err(syn::Error::new(span, "Must be used on a struct")),
    }
    .unwrap_or_else(|err| err.to_compile_error())
    .into()
}
