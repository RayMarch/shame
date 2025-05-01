use std::{marker::PhantomData, num::NonZeroU32};

use super::{
    storage_texture::StorageTexture,
    texture_traits::{
        ChannelFormatShaderType, CubeDir, LayerCoords, RegularGrid, SamplingFormat, Single, StorageTextureCoords,
        StorageTextureFormat, SupportsCoords, TexelShaderType, TextureCoords,
    },
    Texture, TextureKind,
};
use crate::frontend::{any::shared_io::BindPath, rust_types::vec::IsVec};
use crate::frontend::rust_types::vec::vec;
use crate::{
    frontend::{
        any::Any,
        rust_types::{
            index::GpuIndex,
            len::x2,
            reference::{AccessMode, Write},
            vec::{scalar, ToInteger},
            GpuType, To, ToGpuType,
        },
    },
    u32x1,
};

/// (no documentation yet)
pub struct TextureArray<Format, const N: u32, Coords = vec<f32, x2>>
where
    Coords: TextureCoords + LayerCoords,
    Format: SamplingFormat,
{
    any: Any,
    phantom: PhantomData<(Coords, Format)>,
}

/// (no documentation yet)
pub struct StorageTextureArray<Format, const N: u32, Coords = vec<u32, x2>, Access = Write>
where
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Coords: StorageTextureCoords + LayerCoords,
    Access: AccessMode,
{
    any: Any,
    phantom: PhantomData<(Access, Coords, Format)>,
}

impl<Format, const N: u32, Coords> TextureArray<Format, N, Coords>
where
    Format: SamplingFormat + SupportsCoords<Coords>,
    Coords: TextureCoords + LayerCoords,
{
    #[allow(missing_docs)]
    pub const NONZERO_N: NonZeroU32 = match NonZeroU32::new(N) {
        // if your `build` fails here, it means you are instantiating a
        // `TextureArray<_, 0, ..>` type somewhere in your code.
        None => panic!("TextureArrays must have at least one element"),
        Some(n) => n,
    };

    pub(crate) fn from_inner(any: Any) -> Self {
        Self {
            any,
            phantom: PhantomData,
        }
    }

    /// (no documentation yet)
    pub fn at(&self, index: impl ToInteger) -> Texture<Format, Coords, Single> {
        // TODO(release) test the error message that happens when you do at(per_vertex_index) and sample with fragment uvs, and vice versa;
        Texture::from_inner(TextureKind::ArrayLayer {
            texture: self.any,
            layer: index.to_any(),
            shape: Coords::ARRAY_SHAPE(Self::NONZERO_N),
        })
    }

    /// returns the number of layers in this texture array
    pub fn len(&self) -> u32x1 { self.any.texture_num_layers().into() }
}

impl<Format: SamplingFormat, const N: u32> TextureArray<Format, N, vec<f32, x2>> {
    /// (no documentation yet)
    pub fn size(&self) -> vec<u32, x2> { self.any.texture_dimensions(None).into() }

    /// (no documentation yet)
    pub fn size_at_mip_level(&self, level: impl ToInteger) -> vec<u32, x2> {
        self.any.texture_dimensions(Some(level.to_any())).into()
    }

    /// (no documentation yet)
    pub fn mip_level_count(&self) -> u32x1 { self.any.texture_num_levels().into() }
}

impl<Format: SamplingFormat, const N: u32> TextureArray<Format, N, CubeDir> {
    /// size in texels (width, height) of a single cube face,
    /// where width and height are always equal
    pub fn face_size(&self) -> vec<u32, x2> { self.any.texture_dimensions(None).into() }
}

impl<
    Access: AccessMode,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    const N: u32,
    Coords: StorageTextureCoords + LayerCoords,
> StorageTextureArray<Format, N, Coords, Access>
{
    #[allow(missing_docs)]
    pub const NONZERO_N: NonZeroU32 = match NonZeroU32::new(N) {
        // if your `build` fails here, it means you are instantiating a
        // `StorageTextureArray<_, 0, ..>` type somewhere in your code.
        None => panic!("StorageTextureArrays must have a size greater than zero"),
        Some(n) => n,
    };

    pub(crate) fn from_inner(any: Any) -> Self {
        Self {
            any,
            phantom: PhantomData,
        }
    }

    /// (no documentation yet)
    pub fn at(&self, index: impl ToInteger) -> StorageTexture<Format, Coords, Access> {
        // TODO(release) test the error message that happens when you do at(per_vertex_index) and sample with fragment uvs, and vice versa
        StorageTexture::from_inner(TextureKind::ArrayLayer {
            texture: self.any,
            layer: index.to_any(),
            shape: Coords::ARRAY_SHAPE(Self::NONZERO_N),
        })
    }

    // returns the number of layers in this texture array
    /// (no documentation yet)
    pub fn len(&self) -> u32x1 { self.any.texture_num_layers().into() }

    /// (no documentation yet)
    pub fn size(&self) -> vec<u32, Coords::Len>
    where
        Coords: RegularGrid,
    {
        self.any.texture_dimensions(None).into()
    }
}

impl<Idx, Format, const N: u32, Coords> GpuIndex<Idx> for TextureArray<Format, N, Coords>
where
    Idx: ToInteger,
    Format: SamplingFormat + SupportsCoords<Coords>,
    Coords: TextureCoords + LayerCoords,
{
    type Output = Texture<Format, Coords, Single>;

    fn index(&self, index: Idx) -> Texture<Format, Coords, Single> {
        // TODO(release) test the error message that happens when you do at(per_vertex_index) and sample with fragment uvs, and vice versa
        Texture::from_inner(TextureKind::ArrayLayer {
            texture: self.any,
            layer: index.to_any(),
            shape: Coords::ARRAY_SHAPE(Self::NONZERO_N),
        })
    }
}

impl<Idx, Access, Format, const N: u32, Coords> GpuIndex<Idx> for StorageTextureArray<Format, N, Coords, Access>
where
    Idx: ToInteger,
    Access: AccessMode,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Coords: StorageTextureCoords + LayerCoords,
{
    type Output = StorageTexture<Format, Coords, Access>;

    fn index(&self, index: Idx) -> StorageTexture<Format, Coords, Access> { self.at(index) }
}
