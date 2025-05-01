use std::marker::PhantomData;

use super::texture_traits::{RegularGrid, StorageTextureCoords, StorageTextureFormat, SupportsCoords};
use super::TextureKind;
use crate::common::proc_macro_reexports::ir;
use crate::frontend::any::Any;
use crate::frontend::rust_types::len::x2;
use crate::frontend::rust_types::reference::AccessMode;
use crate::frontend::rust_types::reference::{AccessModeReadable, AccessModeWritable, Write};
use crate::frontend::rust_types::vec::{vec, IsVec};
use crate::frontend::rust_types::vec::ToInteger;
use crate::frontend::rust_types::AsAny;
use crate::{u32x1, TextureFormatId};
use crate::Len;

/// (no documentation yet)
pub struct StorageTexture<Format, Coords = vec<u32, x2>, Access = Write>
where
    Coords: StorageTextureCoords,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Access: AccessMode,
{
    inner: TextureKind,
    phantom: PhantomData<(Format, Coords, Access)>,
}

impl<Access, Format, Coords> StorageTexture<Format, Coords, Access>
where
    Coords: StorageTextureCoords,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Access: AccessMode,
{
    pub(crate) fn from_inner(inner: TextureKind) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<Access, Format, Coords> Clone for StorageTexture<Format, Coords, Access>
where
    Coords: StorageTextureCoords,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Access: AccessMode,
{
    fn clone(&self) -> Self { *self }
}

impl<Access, Format, Coords> Copy for StorageTexture<Format, Coords, Access>
where
    Coords: StorageTextureCoords,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Access: AccessMode,
{
}

impl<Format, Coords, Access> StorageTexture<Format, Coords, Access>
where
    Coords: StorageTextureCoords + RegularGrid,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Access: AccessMode,
{
    /// (no documentation yet)
    pub fn size(&self) -> vec<u32, Coords::Len> {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_dimensions(None).into()
    }
}

impl<Access, Format, Coords> StorageTexture<Format, Coords, Access>
where
    Coords: StorageTextureCoords,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    Access: AccessMode,
{
    /// (no documentation yet)
    pub fn store(&self, coords: Coords, value: Format::TexelShaderType)
    where
        Access: AccessModeWritable,
    {
        let (texture, array_index, shape) = match self.inner {
            TextureKind::Standalone(t) => (t, None, Coords::SHAPE),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape,
            } => (t, Some(layer), shape),
        };
        let coords = Coords::texcoord_as_any(coords);
        let value4 = value.as_any().extend_vec_to_len(ir::Len2::X4); // wgsl assumes 4 component writes for every texture format
        texture.texture_store(shape, coords, array_index, value4);
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn load(&self, int_coords: Coords) -> Format::TexelShaderType
    where
        Access: AccessModeReadable,
    {
        let (texture, array_index, shape) = match self.inner {
            TextureKind::Standalone(t) => (t, None, Coords::SHAPE),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape,
            } => (t, Some(layer), shape),
        };
        let coords = Coords::texcoord_as_any(int_coords);
        let any = texture.texture_load(shape, coords, None, array_index);
        let is_depth = Format::id().has_aspect(ir::TextureAspect::Depth);
        match is_depth {
            true => any,
            false => any.vec_x4_shrink(<Format::TexelShaderType as IsVec>::L::LEN),
        }
        .into()
    }
}
