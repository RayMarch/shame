//! /// (no documentation yet)
#![allow(non_snake_case)]
use super::texture_traits::{
    self, Aspect, ColorTargetFormat, Comparison, Multi, Nearest, SamplingFormat, SupportsCoords, SupportsSampler,
    TextureFormat,
};
use super::texture_traits::{Blendable, DepthFormat, DepthStencilFormat, StencilFormat, StorageTextureFormat, SupportsSpp};
use crate::frontend::rust_types::len::Len;
use crate::frontend::rust_types::len::{x1, x2, x3, x4};
use crate::frontend::rust_types::reference::{Read, ReadWrite, Write};
use crate::frontend::rust_types::scalar_type::ScalarType;
use crate::frontend::rust_types::vec::vec;
use crate::ir::{ChannelFormatShaderType, TextureFormatId, TextureSampleUsageType};
use crate::{f32x4, ir};
use astc::*;

#[doc(hidden)]
macro_rules! ignore {
    ($i: ident) => {};
}

enum TextureSampleUsageTypeHelper {
    FilterableFloat,
    NearestFloat,
    NearestUint,
    NearestInt,
    Depth,
}

impl TextureSampleUsageTypeHelper {
    const fn convert(self, len: ir::Len) -> TextureSampleUsageType {
        use TextureSampleUsageTypeHelper as Helper;
        match self {
            Helper::FilterableFloat => TextureSampleUsageType::FilterableFloat { len },
            Helper::NearestFloat => TextureSampleUsageType::Nearest {
                len,
                channel_type: ChannelFormatShaderType::F32,
            },
            Helper::NearestUint => TextureSampleUsageType::Nearest {
                len,
                channel_type: ChannelFormatShaderType::U32,
            },
            Helper::NearestInt => TextureSampleUsageType::Nearest {
                len,
                channel_type: ChannelFormatShaderType::I32,
            },
            Helper::Depth => {
                if let ir::Len::X1 = len {
                    // != is not const, so this is how i write it
                } else {
                    panic!("const panic, depth sample type components != x1")
                }
                TextureSampleUsageType::Depth
            }
        }
    }
}

/// (no documentation yet)
#[rustfmt::skip]
pub mod astc {
    use super::*;
    use std::marker::PhantomData;

    // TODO(docs) put type instantiation examples here. what are valid `B` and `C`?
    /// (no documentation yet)
    pub struct Astc<B: IsAstcBlock, C: IsAstcChannel> {
        phantom: PhantomData<(B, C)>,
    }

    /// (no documentation yet)
    pub trait IsAstcChannel {
        #[allow(missing_docs)]
        const CHANNEL: AstcChannel;
    }

    /// (no documentation yet)
    pub struct Unorm    ; impl IsAstcChannel for Unorm     {const CHANNEL: AstcChannel = AstcChannel::Unorm;}
    /// (no documentation yet)
    pub struct UnormSrgb; impl IsAstcChannel for UnormSrgb {const CHANNEL: AstcChannel = AstcChannel::UnormSrgb;}
    /// (no documentation yet)
    pub struct Hdr      ; impl IsAstcChannel for Hdr       {const CHANNEL: AstcChannel = AstcChannel::Hdr;}

    /// (no documentation yet)
    pub trait IsAstcBlock {
        #[allow(missing_docs)]
        const BLOCK: AstcBlock;
    }

    /// (no documentation yet)
    #[repr(u8)]
    #[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
    pub enum AstcChannel {
        /// (no documentation yet)
        Unorm,
        /// (no documentation yet)
        UnormSrgb,
        /// (no documentation yet)
        Hdr,
    }

    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px).
    pub struct B4x4; impl IsAstcBlock for B4x4 { const BLOCK: AstcBlock = AstcBlock::B4x4;} 
    /// 5x4 block compressed texture. 16 bytes per block (6.4 bit/px).
    pub struct B5x4; impl IsAstcBlock for B5x4 { const BLOCK: AstcBlock = AstcBlock::B5x4;} 
    /// 5x5 block compressed texture. 16 bytes per block (5.12 bit/px).
    pub struct B5x5; impl IsAstcBlock for B5x5 { const BLOCK: AstcBlock = AstcBlock::B5x5;} 
    /// 6x5 block compressed texture. 16 bytes per block (4.27 bit/px).
    pub struct B6x5; impl IsAstcBlock for B6x5 { const BLOCK: AstcBlock = AstcBlock::B6x5;} 
    /// 6x6 block compressed texture. 16 bytes per block (3.56 bit/px).
    pub struct B6x6; impl IsAstcBlock for B6x6 { const BLOCK: AstcBlock = AstcBlock::B6x6;} 
    /// 8x5 block compressed texture. 16 bytes per block (3.2 bit/px).
    pub struct B8x5; impl IsAstcBlock for B8x5 { const BLOCK: AstcBlock = AstcBlock::B8x5;} 
    /// 8x6 block compressed texture. 16 bytes per block (2.67 bit/px).
    pub struct B8x6; impl IsAstcBlock for B8x6 { const BLOCK: AstcBlock = AstcBlock::B8x6;} 
    /// 8x8 block compressed texture. 16 bytes per block (2 bit/px).
    pub struct B8x8; impl IsAstcBlock for B8x8 { const BLOCK: AstcBlock = AstcBlock::B8x8;} 
    /// 10x5 block compressed texture. 16 bytes per block (2.56 bit/px).
    pub struct B10x5; impl IsAstcBlock for B10x5 { const BLOCK: AstcBlock = AstcBlock::B10x5;} 
    /// 10x6 block compressed texture. 16 bytes per block (2.13 bit/px).
    pub struct B10x6; impl IsAstcBlock for B10x6 { const BLOCK: AstcBlock = AstcBlock::B10x6;} 
    /// 10x8 block compressed texture. 16 bytes per block (1.6 bit/px).
    pub struct B10x8; impl IsAstcBlock for B10x8 { const BLOCK: AstcBlock = AstcBlock::B10x8;} 
    /// 10x10 block compressed texture. 16 bytes per block (1.28 bit/px).
    pub struct B10x10; impl IsAstcBlock for B10x10 { const BLOCK: AstcBlock = AstcBlock::B10x10;} 
    /// 12x10 block compressed texture. 16 bytes per block (1.07 bit/px).
    pub struct B12x10; impl IsAstcBlock for B12x10 { const BLOCK: AstcBlock = AstcBlock::B12x10;} 
    /// 12x12 block compressed texture. 16 bytes per block (0.89 bit/px).
    pub struct B12x12; impl IsAstcBlock for B12x12 { const BLOCK: AstcBlock = AstcBlock::B12x12;} 

    /// ASTC block size in texels
    #[repr(u8)]
    #[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
    pub enum AstcBlock {
        /// 4x4 block compressed texture. 16 bytes per block (8 bit/px).
        B4x4,
        /// 5x4 block compressed texture. 16 bytes per block (6.4 bit/px).
        B5x4,
        /// 5x5 block compressed texture. 16 bytes per block (5.12 bit/px).
        B5x5,
        /// 6x5 block compressed texture. 16 bytes per block (4.27 bit/px).
        B6x5,
        /// 6x6 block compressed texture. 16 bytes per block (3.56 bit/px).
        B6x6,
        /// 8x5 block compressed texture. 16 bytes per block (3.2 bit/px).
        B8x5,
        /// 8x6 block compressed texture. 16 bytes per block (2.67 bit/px).
        B8x6,
        /// 8x8 block compressed texture. 16 bytes per block (2 bit/px).
        B8x8,
        /// 10x5 block compressed texture. 16 bytes per block (2.56 bit/px).
        B10x5,
        /// 10x6 block compressed texture. 16 bytes per block (2.13 bit/px).
        B10x6,
        /// 10x8 block compressed texture. 16 bytes per block (1.6 bit/px).
        B10x8,
        /// 10x10 block compressed texture. 16 bytes per block (1.28 bit/px).
        B10x10,
        /// 12x10 block compressed texture. 16 bytes per block (1.07 bit/px).
        B12x10,
        /// 12x12 block compressed texture. 16 bytes per block (0.89 bit/px).
        B12x12,
    }
}

impl<C: IsAstcChannel, B: IsAstcBlock> TextureFormat for Astc<B, C> {
    fn id() -> impl crate::ir::TextureFormatId {
        BuiltinTextureFormatId::Astc {
            channel: C::CHANNEL,
            block: B::BLOCK,
        }
    }
}
impl<C: IsAstcChannel, B: IsAstcBlock> SamplingFormat for Astc<B, C> {
    type SampleType = f32x4;
    const SAMPLE_TYPE: TextureSampleUsageType = TextureSampleUsageType::FilterableFloat { len: ir::Len::X4 };
}
impl<C: IsAstcChannel, B: IsAstcBlock> SupportsSampler<texture_traits::Filtering> for Astc<B, C> {}
impl<C: IsAstcChannel, B: IsAstcBlock> SupportsSampler<texture_traits::Nearest> for Astc<B, C> {}
impl<C: IsAstcChannel, B: IsAstcBlock> Blendable for Astc<B, C> {}
impl<C: IsAstcChannel, B: IsAstcBlock> Aspect for Astc<B, C> {
    type TexelShaderType = f32x4;
}

macro_rules! impl_texture_formats {
    (
        $(
            // Rgba8UnormSrgb
            $Format: ident :
            // (mod rgba, mod float, mod ba)
            ($(
                $(r    $mod_r   : ident)?
                $(rg   $mod_rg  : ident)?
                $(rgb  $mod_rgb : ident)?
                $(rgba $mod_rgba: ident)?
                ,
                $(float $mod_float: ident)?
                $(uint  $mod_uint : ident)?
                $(int   $mod_int  : ident)?
                ,
                $(ba $mod_blend_attachment : ident)?
            )?),
            // SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3>>,
            SupportsCoords<
                $(vec<_ ,  $Coords_x1: ident>)? +
                $(vec<_ ,  $Coords_x2: ident>)? +
                $($CubeDir: ident)? +
                $(vec<_ ,  $Coords_x3: ident>)?
            >
            ,
            // Sampler<Filtering + Nearest + Comparison>,
            $(Sampler< $($Filtering: ident)? + $($Nearest: ident)? + $($Comparison: ident)? >)?,
            // SampleType = vec<FilterableFloat, x4>,
            $(SampleType = vec<$SampleType_Scalar: ident ,  $SampleType_xN: ident> )?,
            // TexelShaderType = vec<f32, x4>,
            $(TexelShaderType = vec<$TexelShaderType_Scalar: ident ,  $TexelShaderType_xN: ident> )?,
            // SupportsSpp<Single + Multi>,
            $(SupportsSpp<Single, $($Multi: ident)?>)?,
            // Storage<Write + Read + ReadWrite>,
            $(Storage<$($Write: ident)? + $($Read: ident)? + $($ReadWrite: ident)?>)?,
            // Blendable,
            $($Blendable: ident)?,
            // Target(Color + Depth + Stencil),
            $(Target($($ColorTgt: ident)? + $($DepthTgt: ident)? + $($StencilTgt: ident)?))?,
            // Aspect< Color( vec<f32, x4> ) + Depth( vec<f32, x1> ) + Stencil( vec<u32, x1> ) >, 
            $(Aspect<
                $( Color  (vec<$Color_Scalar: ident ,    $Color_xN: ident> ) )? +
                $( Depth  (vec<$Depth_Scalar: ident ,    $Depth_xN: ident> ) )? +
                $( Stencil(vec<$Stencil_Scalar: ident ,  $Stencil_xN: ident> ) )?
            >)?,
            // CombinedDepthStencil(Depth24Plus, Stencil8)
            $( CombinedDepthStencil( $DepthPart: ident, $StencilPart: ident ) )?
            ;
        )*
    ) => {
        $(
            #[doc = "A `TextureFormat` type, see https://www.w3.org/TR/webgpu/#texture-formats"]
            pub struct $Format;
        )*
        // define format enum
        #[repr(u8)]
        #[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
        #[doc = "Id of a given builtin texture format"]
        #[allow(missing_docs)]
        pub enum BuiltinTextureFormatId {
            $($Format,)*
            Astc { block: astc::AstcBlock, channel: astc::AstcChannel }
        }

        $(
            impl TextureFormat for $Format {
                fn id() -> impl crate::ir::TextureFormatId {
                    BuiltinTextureFormatId::$Format
                }
            }
        )*

        impl BuiltinTextureFormatId {
            #[doc = "whether a filtering sampler may be used with this texture format"]
            pub fn is_filterable(&self) -> bool {
                match self {
                    $($($($Filtering @ BuiltinTextureFormatId::$Format => true,)?)?)*
                    BuiltinTextureFormatId::Astc {..} => true,
                    _ => false
                }
            }

            #[doc = "whether the format is a color format"]
            pub fn has_color_aspect(&self) -> bool {
                match self {
                    $($($($Color_Scalar @ BuiltinTextureFormatId::$Format => true,)?)?)*
                    BuiltinTextureFormatId::Astc {..} => true,
                    _ => false
                }
            }

            #[doc = "whether the format is a depth format"]
            pub fn has_depth_aspect(&self) -> bool {
                match self {
                    $($($($Depth_Scalar @ BuiltinTextureFormatId::$Format => true,)?)?)*
                    _ => false
                }
            }

            #[doc = "whether the format is a stencil format"]
            pub fn has_stencil_aspect(&self) -> bool {
                match self {
                    $($($($Stencil_Scalar @ BuiltinTextureFormatId::$Format => true,)?)?)*
                    _ => false
                }
            }

            pub(crate) fn is_blendable_impl(&self) -> bool {
                match self {
                    $($($Blendable @ BuiltinTextureFormatId::$Format => true,)?)*
                    _ => false
                }
            }

            /// the type that appears in the shader after sampling from this texture
            /// 
            /// `None` for non sampleable textures
            pub fn sampling_result_type(&self) -> Option<(crate::ir::Len, crate::ir::ChannelFormatShaderType)> {
                #[allow(non_camel_case_types)]
                enum Scalar {f32, i32, u32}
                use crate::{frontend::rust_types::len::Len, ir::ChannelFormatShaderType};
                let option_len_scal = match self {
                    $($(
                        BuiltinTextureFormatId::$Format => Some((<$TexelShaderType_xN as Len>::LEN, Scalar:: $TexelShaderType_Scalar)),
                        BuiltinTextureFormatId::Astc {..} => Some((<x4 as Len>::LEN, Scalar::f32)),
                    )?)*
                    _ => None
                };
                option_len_scal.map(|(len, scalar)| (len,
                    match scalar {
                        Scalar::i32 => ChannelFormatShaderType::I32,
                        Scalar::u32 => ChannelFormatShaderType::U32,
                        Scalar::f32 => ChannelFormatShaderType::F32,
                    }
                ))
            }
        }

        $($(
            impl<S: ScalarType> SupportsCoords<vec<S, $Coords_x1>> for $Format {}
        )?)*
        // Coords_x2 and CubeDir are handled by a blanket implementation
        $($(
            impl<S: ScalarType> SupportsCoords<vec<S, $Coords_x3>> for $Format {}
        )?)*

        $($($(
            impl SupportsSampler<texture_traits:: $Filtering> for $Format {}
        )?)?)*

        $($($(
            impl SupportsSampler<$Nearest> for $Format {}
        )?)?)*

        $($($(
            impl SupportsSampler<$Comparison> for $Format {}
        )?)?)*

        $($(
            impl SamplingFormat for $Format {
                type SampleType = vec<$TexelShaderType_Scalar, $TexelShaderType_xN>;
                const SAMPLE_TYPE: TextureSampleUsageType = TextureSampleUsageTypeHelper:: $SampleType_Scalar .convert(<$SampleType_xN as Len>::LEN);
            }

            impl Aspect for $Format {
                type TexelShaderType = vec<$TexelShaderType_Scalar, $TexelShaderType_xN>;
            }
        )?)*

        $($($(
            impl StorageTextureFormat<$Write> for $Format {}
        )?)?)*

        $($($(
            impl StorageTextureFormat<$Read> for $Format {}
        )?)?)*

        $($($(
            impl StorageTextureFormat<$ReadWrite> for $Format {}
        )?)?)*

        $($($(
            impl SupportsSpp<$Multi> for $Format {}
        )?)?)*

        $($(
            impl $Blendable for $Format {}
        )?)*

        $($($(
            ignore!{$ColorTgt}
            impl ColorTargetFormat for $Format {}
        )?)?)*

        $($($(
            impl DepthFormat for $Format {
                type DepthShaderType = vec<$Depth_Scalar, $Depth_xN>;
            }
        )?)?)*

        $($($(
            impl StencilFormat for $Format {
                type StencilShaderType = vec<$Stencil_Scalar, $Stencil_xN>;
            }
        )?)?)*

        $($(
            impl DepthStencilFormat for $Format {
                type Depth = $DepthPart;
                type Stencil = $StencilPart;
            }
        )?)*

        #[doc = "all texture formats, categorized in modules with regards to the traits they implement"]
        pub mod browse {
            #[doc = "color formats that return `vec<_, x1>` when sampled"]
            pub mod r             {$($($(ignore!{$mod_r    } #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "color formats that return `vec<_, x2>` when sampled"]
            pub mod rg            {$($($(ignore!{$mod_rg   } #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "color formats that return `vec<_, x3>` when sampled"]
            pub mod rgb           {$($($(ignore!{$mod_rgb  } #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "color formats that return `vec<_, x4>` when sampled"]
            pub mod rgba          {
                $($($(ignore!{$mod_rgba } #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*
                type Astc<AstcChannel, AstcBlock> = crate::texture_formats::astc::Astc<AstcChannel, AstcBlock>;
            }
            #[doc = "formats that return `vec<i32, _>` when sampled"]
            pub mod int           {$($($(ignore!{$mod_int } #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "formats that return `vec<u32, _>` when sampled"]
            pub mod uint          {$($($(ignore!{$mod_uint } #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "formats that return `vec<f32, _>` when sampled"]
            pub mod float         {
                $($($(ignore!{$mod_float} #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*
                type Astc<AstcChannel, AstcBlock> = crate::texture_formats::astc::Astc<AstcChannel, AstcBlock>;
            }
            #[doc = "implements `ColorTargetFormat`"]
            pub mod color_target  {$($($(ignore!{$ColorTgt}      #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "implements `DepthFormat`"]
            pub mod depth         {$($($(ignore!{$DepthTgt}      #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "implements `StencilFormat`"]
            pub mod stencil       {$($($(ignore!{$StencilTgt}    #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "implements `StorageTextureFormat<Write>`"]
            pub mod storage_write {$($($(ignore!{$Write}         #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
            #[doc = "implements `SupportsSpp<Multi>`"]
            pub mod multisample   {$($($(  ignore!{$Multi}       #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format; )?)?)*}
            #[doc = "implements `SupportsSampler<Filterable>`"]
            pub mod filterable    {
                $($($(ignore!{$Filtering}     #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*
                type Astc<AstcChannel, AstcBlock> = crate::texture_formats::astc::Astc<AstcChannel, AstcBlock>;
            }
            #[doc = "implements `ColorTargetFormat` + `Blendable`"]
            pub mod blendable {$($($(ignore!{$mod_blend_attachment} #[allow(missing_docs)] pub type $Format = crate::texture_formats::$Format;)?)?)*}
        }
    };
}

// This table was generated by `texture_format_table_generator.rs` using `wgpu`
impl_texture_formats! {
    R8Unorm             : (r    mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>,                                  , Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    R8Snorm             : (r    mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    R8Uint              : (r    mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x1>, TexelShaderType = vec<u32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<u32, x1>) +                     +                      >, ;
    R8Sint              : (r    mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x1>, TexelShaderType = vec<i32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<i32, x1>) +                     +                      >, ;
    R16Uint             : (r    mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x1>, TexelShaderType = vec<u32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<u32, x1>) +                     +                      >, ;
    R16Sint             : (r    mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x1>, TexelShaderType = vec<i32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<i32, x1>) +                     +                      >, ;
    R16Unorm            : (r    mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable,                                , Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    R16Snorm            : (r    mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable,                                , Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    R16Float            : (r    mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>,                                  , Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    Rg8Unorm            : (rg   mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single, Multi>,                                  , Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Rg8Snorm            : (rg   mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Rg8Uint             : (rg   mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x2>, TexelShaderType = vec<u32, x2>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<u32, x2>) +                     +                      >, ;
    Rg8Sint             : (rg   mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x2>, TexelShaderType = vec<i32, x2>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<i32, x2>) +                     +                      >, ;
    R32Uint             : (r    mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x1>, TexelShaderType = vec<u32, x1>, SupportsSpp<Single,      >, Storage<Write + Read + ReadWrite>,          , Target(Color +       +        ), Aspect<Color(vec<u32, x1>) +                     +                      >, ;
    R32Sint             : (r    mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x1>, TexelShaderType = vec<i32, x1>, SupportsSpp<Single,      >, Storage<Write + Read + ReadWrite>,          , Target(Color +       +        ), Aspect<Color(vec<i32, x1>) +                     +                      >, ;
    R32Float            : (r    mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>, Storage<Write + Read + ReadWrite>,          , Target(Color +       +        ), Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    Rg16Uint            : (rg   mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x2>, TexelShaderType = vec<u32, x2>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<u32, x2>) +                     +                      >, ;
    Rg16Sint            : (rg   mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x2>, TexelShaderType = vec<i32, x2>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<i32, x2>) +                     +                      >, ;
    Rg16Unorm           : (rg   mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable,                                , Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Rg16Snorm           : (rg   mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable,                                , Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Rg16Float           : (rg   mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single, Multi>,                                  , Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Rgba8Unorm          : (rgba mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rgba8UnormSrgb      : (rgba mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>,                                  , Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rgba8Snorm          : (rgba mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >, Storage<Write +      +          >, Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rgba8Uint           : (rgba mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x4>, TexelShaderType = vec<u32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<u32, x4>) +                     +                      >, ;
    Rgba8Sint           : (rgba mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x4>, TexelShaderType = vec<i32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<i32, x4>) +                     +                      >, ;
    Bgra8Unorm          : (rgba mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bgra8UnormSrgb      : (rgba mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>,                                  , Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rgb9e5Ufloat        : (rgb  mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x3>, TexelShaderType = vec<f32, x3>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x3>) +                     +                      >, ;
    Rgb10a2Uint         : (rgba mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x4>, TexelShaderType = vec<u32, x4>, SupportsSpp<Single, Multi>,                                  ,          , Target(Color +       +        ), Aspect<Color(vec<u32, x4>) +                     +                      >, ;
    Rgb10a2Unorm        : (rgba mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>,                                  , Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rg32Uint            : (rg   mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x2>, TexelShaderType = vec<u32, x2>, SupportsSpp<Single,      >, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<u32, x2>) +                     +                      >, ;
    Rg32Sint            : (rg   mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x2>, TexelShaderType = vec<i32, x2>, SupportsSpp<Single,      >, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<i32, x2>) +                     +                      >, ;
    Rg32Float           : (rg   mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single,      >, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Rgba16Uint          : (rgba mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x4>, TexelShaderType = vec<u32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<u32, x4>) +                     +                      >, ;
    Rgba16Sint          : (rgba mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x4>, TexelShaderType = vec<i32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<i32, x4>) +                     +                      >, ;
    Rgba16Unorm         : (rgba mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rgba16Snorm         : (rgba mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rgba16Float         : (rgba mod, float mod, ba mod), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single, Multi>, Storage<Write +      +          >, Blendable, Target(Color +       +        ), Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Rgba32Uint          : (rgba mod, uint  mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x4>, TexelShaderType = vec<u32, x4>, SupportsSpp<Single,      >, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<u32, x4>) +                     +                      >, ;
    Rgba32Sint          : (rgba mod, int   mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<          + Nearest +           >, SampleType = vec<NearestInt     , x4>, TexelShaderType = vec<i32, x4>, SupportsSpp<Single,      >, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<i32, x4>) +                     +                      >, ;
    Rgba32Float         : (rgba mod, float mod,       ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >, Storage<Write +      +          >,          , Target(Color +       +        ), Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Stencil8            : (                           ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<          + Nearest +           >, SampleType = vec<NearestUint    , x1>, TexelShaderType = vec<u32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(      +       + Stencil), Aspect<                    +                     + Stencil(vec<u32, x1>)>, ;
    Depth16Unorm        : (                           ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<          + Nearest + Comparison>, SampleType = vec<Depth          , x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(      + Depth +        ), Aspect<                    + Depth(vec<f32, x1>) +                      >, ;
    Depth24Plus         : (                           ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<          + Nearest + Comparison>, SampleType = vec<Depth          , x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(      + Depth +        ), Aspect<                    + Depth(vec<f32, x1>) +                      >, ;
    Depth24PlusStencil8 : (                           ), SupportsCoords<           + vec<_, x2> + CubeDir +            >,                                          ,                                      ,                               , SupportsSpp<Single, Multi>,                                  ,          , Target(      + Depth + Stencil), Aspect<                    + Depth(vec<f32, x1>) + Stencil(vec<u32, x1>)>, CombinedDepthStencil(Depth24Plus, Stencil8);
    Depth32Float        : (                           ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<          + Nearest + Comparison>, SampleType = vec<Depth          , x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single, Multi>,                                  ,          , Target(      + Depth +        ), Aspect<                    + Depth(vec<f32, x1>) +                      >, ;
    Depth32FloatStencil8: (                           ), SupportsCoords<           + vec<_, x2> + CubeDir +            >,                                          ,                                      ,                               , SupportsSpp<Single, Multi>,                                  ,          , Target(      + Depth + Stencil), Aspect<                    + Depth(vec<f32, x1>) + Stencil(vec<u32, x1>)>, CombinedDepthStencil(Depth32Float, Stencil8);
    NV12                : (                           ), SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3> >,                                          ,                                      ,                               , SupportsSpp<Single,      >,                                  ,          ,                                ,                                                                          , ;
    Bc1RgbaUnorm        : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bc1RgbaUnormSrgb    : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bc2RgbaUnorm        : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bc2RgbaUnormSrgb    : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bc3RgbaUnorm        : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bc3RgbaUnormSrgb    : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bc4RUnorm           : (r    mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    Bc4RSnorm           : (r    mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    Bc5RgUnorm          : (rg   mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Bc5RgSnorm          : (rg   mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    Bc6hRgbUfloat       : (rgb  mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x3>, TexelShaderType = vec<f32, x3>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x3>) +                     +                      >, ;
    Bc6hRgbFloat        : (rgb  mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x3>, TexelShaderType = vec<f32, x3>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x3>) +                     +                      >, ;
    Bc7RgbaUnorm        : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Bc7RgbaUnormSrgb    : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Etc2Rgb8Unorm       : (rgb  mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x3>, TexelShaderType = vec<f32, x3>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x3>) +                     +                      >, ;
    Etc2Rgb8UnormSrgb   : (rgb  mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x3>, TexelShaderType = vec<f32, x3>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x3>) +                     +                      >, ;
    Etc2Rgb8A1Unorm     : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Etc2Rgb8A1UnormSrgb : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Etc2Rgba8Unorm      : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    Etc2Rgba8UnormSrgb  : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
    EacR11Unorm         : (r    mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    EacR11Snorm         : (r    mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x1>, TexelShaderType = vec<f32, x1>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x1>) +                     +                      >, ;
    EacRg11Unorm        : (rg   mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x2>) +                     +                      >, ;
    EacRg11Snorm        : (rg   mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +            >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x2>, TexelShaderType = vec<f32, x2>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x2>) +                     +                      >, ;

}
//  astc::Astc<B, C>    : (rgba mod, float mod,       ), SupportsCoords<           + vec<_, x2> + CubeDir +           >, Sampler<Filtering + Nearest +           >, SampleType = vec<FilterableFloat, x4>, TexelShaderType = vec<f32, x4>, SupportsSpp<Single,      >,                                  , Blendable,                                , Aspect<Color(vec<f32, x4>) +                     +                      >, ;
// (astc::Astc is implemented separately because of its generic arguments)
