use crate::frontend::any::Any;
use crate::frontend::any::{shared_io::BufferBindingType, InvalidReason};
use crate::frontend::rust_types::layout_traits::GpuLayout;
use crate::frontend::rust_types::reference::Ref;
use crate::frontend::rust_types::reference::{AccessMode, AccessModeReadable, Read};
use crate::frontend::rust_types::type_traits::BindingArgs;
use crate::frontend::rust_types::{reference::ReadWrite, struct_::SizedFields, type_traits::NoBools};
use crate::frontend::texture::storage_texture::StorageTexture;
use crate::frontend::texture::texture_array::{StorageTextureArray, TextureArray};
use crate::frontend::texture::texture_traits::{SamplingFormat, Spp, StorageTextureFormat};
use crate::frontend::texture::{Sampler, Texture};
use crate::ir::pipeline::StageMask;
use crate::{
    frontend::any::{shared_io::BindPath, shared_io::BindingType},
    frontend::{
        rust_types::reference::AccessModeWritable,
        texture::{
            texture_traits::{
                LayerCoords, SamplingMethod, StorageTextureCoords, SupportsCoords, SupportsSpp, TextureCoords,
            },
            TextureKind,
        },
    },
    ir::{self, TextureFormatWrapper},
    mem::{self, AddressSpace, SupportsAccess},
};
use std::marker::PhantomData;
use std::num::NonZeroU32;

/// Types of resources that can be bound as part of a bind-group and accessed in Gpu pipelines.
///
/// [`Binding`] types include
/// - [`Buffer`] (readonly)
/// - [`BufferRef`] (supports readwrite and atomics)
/// - [`Sampler`]
/// - [`Texture`] (for sampling)
/// - [`StorageTexture`] (for writing)
///
/// [`Buffer`]: crate::Buffer
/// [`BufferRef`]: crate::BufferRef
#[diagnostic::on_unimplemented(message = "`{Self}` is not a valid type for a bind-group binding")]
pub trait Binding {
    /// runtime representation of `Self`
    fn binding_type() -> BindingType;

    /// shader type of `Self`
    fn store_ty() -> ir::StoreType;

    #[doc(hidden)]
    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self;
}

impl<Format, Coords, SPP> Binding for Texture<Format, Coords, SPP>
where
    Coords: TextureCoords + SupportsSpp<SPP>,
    Format: SamplingFormat + SupportsSpp<SPP> + SupportsCoords<Coords>,
    SPP: Spp,
{
    fn binding_type() -> BindingType {
        BindingType::SampledTexture {
            shape: Coords::SHAPE,
            sample_type: Format::SAMPLE_TYPE.restrict_with_spp(SPP::SAMPLES_PER_PIXEL),
            samples_per_pixel: SPP::SAMPLES_PER_PIXEL,
        }
    }

    fn store_ty() -> ir::StoreType {
        let shape = Coords::SHAPE;
        let sample_type = Format::SAMPLE_TYPE;
        let spp = SPP::SAMPLES_PER_PIXEL;
        ir::StoreType::Handle(ir::HandleType::SampledTexture(
            shape,
            sample_type.restrict_with_spp(spp),
            spp,
        ))
    }

    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self {
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => {
                Any::binding(path, visibility, Self::store_ty(), Self::binding_type(), false)
            }
        };
        Texture::from_inner(TextureKind::Standalone(any))
    }
}

impl<Access: AccessMode, Format: StorageTextureFormat<Access> + SupportsCoords<Coords>, Coords: StorageTextureCoords>
    Binding for StorageTexture<Format, Coords, Access>
{
    fn binding_type() -> BindingType {
        BindingType::StorageTexture {
            shape: Coords::SHAPE,
            format: Format::id().into(),
            access: Access::ACCESS,
        }
    }

    fn store_ty() -> ir::StoreType {
        let shape = Coords::SHAPE;
        let access = Access::ACCESS;
        let format = TextureFormatWrapper::new(Format::id());
        ir::StoreType::Handle(ir::HandleType::StorageTexture(shape, format, access))
    }

    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self {
        let shape = Coords::SHAPE;
        let access = Access::ACCESS;
        let format = TextureFormatWrapper::new(Format::id());
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => {
                Any::binding(path, visibility, Self::store_ty(), Self::binding_type(), false)
            }
        };
        StorageTexture::from_inner(TextureKind::Standalone(any))
    }
}

impl<Format, const N: u32, Coords> Binding for TextureArray<Format, N, Coords>
where
    Format: SamplingFormat + SupportsCoords<Coords>,
    Coords: TextureCoords + LayerCoords,
{
    fn binding_type() -> BindingType {
        let samples_per_pixel = ir::SamplesPerPixel::Single;
        BindingType::SampledTexture {
            shape: Coords::ARRAY_SHAPE(Self::NONZERO_N),
            sample_type: Format::SAMPLE_TYPE.restrict_with_spp(samples_per_pixel),
            samples_per_pixel,
        }
    }

    fn store_ty() -> ir::StoreType {
        let shape = Coords::ARRAY_SHAPE(Self::NONZERO_N);
        let sample_type = Format::SAMPLE_TYPE;
        let spp = ir::SamplesPerPixel::Single;
        ir::StoreType::Handle(ir::HandleType::SampledTexture(
            shape,
            sample_type.restrict_with_spp(spp),
            spp,
        ))
    }

    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self {
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => {
                Any::binding(path, visibility, Self::store_ty(), Self::binding_type(), false)
            }
        };
        TextureArray::from_inner(any)
    }
}

impl<
    Access: AccessMode,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    const N: u32,
    Coords: StorageTextureCoords + LayerCoords,
> Binding for StorageTextureArray<Format, N, Coords, Access>
{
    fn binding_type() -> BindingType {
        BindingType::StorageTexture {
            shape: Coords::ARRAY_SHAPE(Self::NONZERO_N),
            format: Format::id().into(),
            access: Access::ACCESS,
        }
    }

    fn store_ty() -> ir::StoreType {
        let shape = Coords::ARRAY_SHAPE(Self::NONZERO_N);
        let access = Access::ACCESS;
        let format = TextureFormatWrapper::new(Format::id());
        ir::StoreType::Handle(ir::HandleType::StorageTexture(shape, format, access))
    }

    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self {
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => {
                Any::binding(path, visibility, Self::store_ty(), Self::binding_type(), false)
            }
        };
        StorageTextureArray::from_inner(any)
    }
}

impl<M: SamplingMethod> Binding for Sampler<M> {
    fn binding_type() -> BindingType { BindingType::Sampler(M::SAMPLING_METHOD) }

    fn store_ty() -> ir::StoreType { ir::StoreType::Handle(ir::HandleType::Sampler(M::SAMPLING_METHOD)) }

    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self {
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => Any::binding(
                path,
                visibility,
                ir::StoreType::Handle(ir::HandleType::Sampler(M::SAMPLING_METHOD)),
                Self::binding_type(),
                false,
            ),
        };
        Self::from_inner(any)
    }
}
