#![allow(non_camel_case_types, clippy::eq_op)]
use std::marker::PhantomData;

use crate::{frontend::rust_types::{len::*, vec::vec, scalar_type::ScalarType, vec::IsVec}, ir, Value};

use super::texture_traits::*;
use crate::common::floating_point::f16;

/// unsigned normalized 8 bit channel
pub struct unorm8;
/// unsigned normalized 16 bit channel
pub struct unorm16;
/// unsigned normalized 24 bit channel (used for depth buffers)
pub struct unorm24;
/// unsigned normalized 24 or 32 bit channel (used for depth buffers)
pub struct unorm24plus;

/// signed normalized 8 bit channel
pub struct snorm8;
/// unsigned normalized 16 bit channel
pub struct snorm16;

#[rustfmt::skip] impl ChannelFormat for unorm8 { const BYTE_SIZE: u32 =  8/8; type ShaderType = f32;}
#[rustfmt::skip] impl ChannelFormat for snorm8 { const BYTE_SIZE: u32 =  8/8; type ShaderType = f32;}
#[rustfmt::skip] impl ChannelFormat for u8     { const BYTE_SIZE: u32 =  8/8; type ShaderType = u32;}
#[rustfmt::skip] impl ChannelFormat for i8     { const BYTE_SIZE: u32 =  8/8; type ShaderType = i32;}
#[rustfmt::skip] impl ChannelFormat for u16    { const BYTE_SIZE: u32 = 16/8; type ShaderType = u32;}
#[rustfmt::skip] impl ChannelFormat for i16    { const BYTE_SIZE: u32 = 16/8; type ShaderType = i32;}
#[rustfmt::skip] impl ChannelFormat for f16    { const BYTE_SIZE: u32 = 16/8; type ShaderType = f32;}
#[rustfmt::skip] impl ChannelFormat for u32    { const BYTE_SIZE: u32 = 32/8; type ShaderType = u32;}
#[rustfmt::skip] impl ChannelFormat for i32    { const BYTE_SIZE: u32 = 32/8; type ShaderType = i32;}
#[rustfmt::skip] impl ChannelFormat for f32    { const BYTE_SIZE: u32 = 32/8; type ShaderType = f32;}

#[allow(non_camel_case_types)]
pub struct sRGB;
pub struct Linear;

pub trait ColorSpace {}
impl ColorSpace for sRGB {}
impl ColorSpace for Linear {}

pub struct R;
pub struct Rg;
pub struct Rgb;
pub struct Rgba;
pub struct Bgra;

pub trait ColorLen {}
impl ColorLen for R {}
impl ColorLen for Rg {}
impl ColorLen for Rgb {}
impl ColorLen for Rgba {}
impl ColorLen for Bgra {}


pub trait DepthChannelFormat {}
impl DepthChannelFormat for unorm16 {}
impl DepthChannelFormat for unorm24 {}
impl DepthChannelFormat for unorm24plus {}
impl DepthChannelFormat for f32 {}

pub trait StencilChannelFormat {}
impl StencilChannelFormat for u8 {}

pub struct DepthStencilFmt<Depth, Stencil = ()>(PhantomData<(Depth, Stencil)>);
pub struct ColorFmt<Rgba: ColorLen, F: ChannelFormat, S: ColorSpace = Linear>(PhantomData<(Rgba, F, S)>);

// Test formats (= Depth and stencil formats)
pub type Stencil8             = DepthStencilFmt<(), u8>;
pub type Depth16Unorm         = DepthStencilFmt<unorm16>;
pub type Depth24Plus          = DepthStencilFmt<unorm24plus>;
pub type Depth24PlusStencil8  = DepthStencilFmt<unorm24plus, u8>;
pub type Depth32Float         = DepthStencilFmt<f32>;
pub type Depth32FloatStencil8 = DepthStencilFmt<f32, u8>;

// Normal 8 bit formats
pub type R8Unorm = ColorFmt<R, unorm8>;
pub type R8Snorm = ColorFmt<R, snorm8>;
pub type R8Uint  = ColorFmt<R, u8>;
pub type R8Sint  = ColorFmt<R, i8>;

// Normal 16 bit formats
pub type R16Uint  = ColorFmt<R , u16>;
pub type R16Sint  = ColorFmt<R , i16>;
pub type R16Unorm = ColorFmt<R , unorm16>;
pub type R16Snorm = ColorFmt<R , snorm16>;
pub type R16Float = ColorFmt<R , f16>;
pub type Rg8Unorm = ColorFmt<Rg, unorm8>;
pub type Rg8Snorm = ColorFmt<Rg, snorm8>;
pub type Rg8Uint  = ColorFmt<Rg, u8>;
pub type Rg8Sint  = ColorFmt<Rg, i8>;

// Normal 32 bit formats
pub type R32Uint        = ColorFmt<R   , u32>;
pub type R32Sint        = ColorFmt<R   , i32>;
pub type R32Float       = ColorFmt<R   , f32>;
pub type Rg16Uint       = ColorFmt<Rg  , u16>;
pub type Rg16Sint       = ColorFmt<Rg  , i16>;
pub type Rg16Unorm      = ColorFmt<Rg  , unorm16>;
pub type Rg16Snorm      = ColorFmt<Rg  , snorm16>;
pub type Rg16Float      = ColorFmt<Rg  , f16>;
pub type Rgba8Unorm     = ColorFmt<Rgba, unorm8>;
pub type Rgba8UnormSrgb = ColorFmt<Rgba, unorm8, sRGB>;
pub type Rgba8Snorm     = ColorFmt<Rgba, snorm8>;
pub type Rgba8Uint      = ColorFmt<Rgba, u8>;
pub type Rgba8Sint      = ColorFmt<Rgba, i8>;
pub type Bgra8Unorm     = ColorFmt<Bgra, unorm8>;
pub type Bgra8UnormSrgb = ColorFmt<Bgra, unorm8, sRGB>;

// Normal 64 bit formats
pub type Rg32Uint      = ColorFmt<Rg  , u32>;
pub type Rg32Sint      = ColorFmt<Rg  , i32>;
pub type Rg32Float     = ColorFmt<Rg  , f32>;
pub type Rgba16Uint    = ColorFmt<Rgba, u16>;
pub type Rgba16Sint    = ColorFmt<Rgba, i16>;
pub type Rgba16Unorm   = ColorFmt<Rgba, unorm16>;
pub type Rgba16Snorm   = ColorFmt<Rgba, snorm16>;
pub type Rgba16Float   = ColorFmt<Rgba, f16>;

// Normal 128 bit formats
pub type Rgba32Uint    = ColorFmt<Rgba, u32>;
pub type Rgba32Sint    = ColorFmt<Rgba, i32>;
pub type Rgba32Float   = ColorFmt<Rgba, f32>;

// Packed 32 bit formats
pub struct Rgb9e5Ufloat;
pub struct Rgb10a2Uint ;
pub struct Rgb10a2Unorm;
pub struct Rg11b10Float;

pub trait CompressedChannelFormat {}
pub struct  unorm_compressed;
pub struct  snorm_compressed;
pub struct ufloat_compressed;
pub struct sfloat_compressed;
impl CompressedChannelFormat for  unorm_compressed {}
impl CompressedChannelFormat for  snorm_compressed {}
impl CompressedChannelFormat for ufloat_compressed {}
impl CompressedChannelFormat for sfloat_compressed {}

/// "BC" (Block Compression) texture format
pub mod bc {
    use super::*;

    pub trait BcVersion {}
    pub struct Bc1;  impl BcVersion for Bc1 {}
    pub struct Bc2;  impl BcVersion for Bc2 {}
    pub struct Bc3;  impl BcVersion for Bc3 {}
    pub struct Bc4;  impl BcVersion for Bc4 {}
    pub struct Bc5;  impl BcVersion for Bc5 {}
    pub struct Bc6h; impl BcVersion for Bc6h {}
    pub struct Bc7;  impl BcVersion for Bc7 {}

    pub struct ColorFmt<Ver: BcVersion, Rgba: ColorLen, F: CompressedChannelFormat, S: ColorSpace = Linear>
    (PhantomData<(Ver, Rgba, F, S)>);
}

use bc::*;
pub type Bc1RgbaUnorm      = bc::ColorFmt<Bc1 , Rgba, unorm_compressed>;
pub type Bc1RgbaUnormSrgb  = bc::ColorFmt<Bc1 , Rgba, unorm_compressed, sRGB>;
pub type Bc2RgbaUnorm      = bc::ColorFmt<Bc2 , Rgba, unorm_compressed>;
pub type Bc2RgbaUnormSrgb  = bc::ColorFmt<Bc2 , Rgba, unorm_compressed, sRGB>;
pub type Bc3RgbaUnorm      = bc::ColorFmt<Bc3 , Rgba, unorm_compressed>;
pub type Bc3RgbaUnormSrgb  = bc::ColorFmt<Bc3 , Rgba, unorm_compressed, sRGB>;
pub type Bc4RUnorm         = bc::ColorFmt<Bc4 , R   , unorm_compressed>;
pub type Bc4RSnorm         = bc::ColorFmt<Bc4 , R   , snorm_compressed>;
pub type Bc5RgUnorm        = bc::ColorFmt<Bc5 , Rg  , unorm_compressed>;
pub type Bc5RgSnorm        = bc::ColorFmt<Bc5 , Rg  , snorm_compressed>;
pub type Bc6hRgbUfloat     = bc::ColorFmt<Bc6h, Rgb , ufloat_compressed>;
pub type Bc6hRgbFloat      = bc::ColorFmt<Bc6h, Rgb , sfloat_compressed>;
pub type Bc7RgbaUnorm      = bc::ColorFmt<Bc7 , Rgba, unorm_compressed>;
pub type Bc7RgbaUnormSrgb  = bc::ColorFmt<Bc7 , Rgba, unorm_compressed, sRGB>;

/// Ericsson Texture Compression 2
pub mod etc2 {
    use super::*;

    pub trait Etc2Channels {}
    pub struct Rgb8  ; impl Etc2Channels for Rgb8   {}
    pub struct Rgb8A1; impl Etc2Channels for Rgb8A1 {}
    pub struct Rgba8 ; impl Etc2Channels for Rgba8  {}
    
    pub struct ColorFmt<Rgba: Etc2Channels, F: CompressedChannelFormat, S: ColorSpace = Linear>
    (PhantomData<(Rgba, F, S)>);
}

use etc2::*;
pub type Etc2Rgb8Unorm       = etc2::ColorFmt<Rgb8  , unorm_compressed>;
pub type Etc2Rgb8UnormSrgb   = etc2::ColorFmt<Rgb8  , unorm_compressed, sRGB>;
pub type Etc2Rgb8A1Unorm     = etc2::ColorFmt<Rgb8A1, unorm_compressed>;
pub type Etc2Rgb8A1UnormSrgb = etc2::ColorFmt<Rgb8A1, unorm_compressed, sRGB>;
pub type Etc2Rgba8Unorm      = etc2::ColorFmt<Rgba8 , unorm_compressed>;
pub type Etc2Rgba8UnormSrgb  = etc2::ColorFmt<Rgba8 , unorm_compressed, sRGB>;

pub mod eac {
    use super::*;

    pub trait EacChannels {}
    pub struct R11 ; impl EacChannels for R11  {}
    pub struct Rg11; impl EacChannels for Rg11 {}
    
    pub struct ColorFmt<Rg: EacChannels, F: CompressedChannelFormat, S: ColorSpace = Linear>
    (PhantomData<(Rg, F, S)>);
}

use eac::*;
pub type EacR11Unorm  = eac::ColorFmt<R11 , unorm_compressed>;
pub type EacR11Snorm  = eac::ColorFmt<R11 , snorm_compressed>;
pub type EacRg11Unorm = eac::ColorFmt<Rg11, unorm_compressed>;
pub type EacRg11Snorm = eac::ColorFmt<Rg11, snorm_compressed>;

pub mod astc {
    use super::*;

    pub trait AstcBlockSize {}

    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px).
    pub struct B4x4; impl AstcBlockSize for B4x4 {}  
    /// 5x4 block compressed texture. 16 bytes per block (6.4 bit/px).
    pub struct B5x4; impl AstcBlockSize for B5x4 {} 
    /// 5x5 block compressed texture. 16 bytes per block (5.12 bit/px).
    pub struct B5x5; impl AstcBlockSize for B5x5 {} 
    /// 6x5 block compressed texture. 16 bytes per block (4.27 bit/px).
    pub struct B6x5; impl AstcBlockSize for B6x5 {} 
    /// 6x6 block compressed texture. 16 bytes per block (3.56 bit/px).
    pub struct B6x6; impl AstcBlockSize for B6x6 {} 
    /// 8x5 block compressed texture. 16 bytes per block (3.2 bit/px).
    pub struct B8x5; impl AstcBlockSize for B8x5 {} 
    /// 8x6 block compressed texture. 16 bytes per block (2.67 bit/px).
    pub struct B8x6; impl AstcBlockSize for B8x6 {} 
    /// 8x8 block compressed texture. 16 bytes per block (2 bit/px).
    pub struct B8x8; impl AstcBlockSize for B8x8 {} 
    /// 10x5 block compressed texture. 16 bytes per block (2.56 bit/px).
    pub struct B10x5; impl AstcBlockSize for B10x5 {} 
    /// 10x6 block compressed texture. 16 bytes per block (2.13 bit/px).
    pub struct B10x6; impl AstcBlockSize for B10x6 {} 
    /// 10x8 block compressed texture. 16 bytes per block (1.6 bit/px).
    pub struct B10x8; impl AstcBlockSize for B10x8 {} 
    /// 10x10 block compressed texture. 16 bytes per block (1.28 bit/px).
    pub struct B10x10; impl AstcBlockSize for B10x10 {} 
    /// 12x10 block compressed texture. 16 bytes per block (1.07 bit/px).
    pub struct B12x10; impl AstcBlockSize for B12x10 {} 
    /// 12x12 block compressed texture. 16 bytes per block (0.89 bit/px).
    pub struct B12x12; impl AstcBlockSize for B12x12 {} 

    pub struct ColorFmt<B: AstcBlockSize, Rgba: ColorLen, F: CompressedChannelFormat, S: ColorSpace = Linear>
    (PhantomData<(B, Rgba, F, S)>);
}

type AstcUnorm    <AstcBlock> = astc::ColorFmt<AstcBlock, Rgba,  unorm_compressed, Linear>;
type AstcUnormSrgb<AstcBlock> = astc::ColorFmt<AstcBlock, Rgba,  unorm_compressed, sRGB>;
type AstcHdr      <AstcBlock> = astc::ColorFmt<AstcBlock, Rgba, ufloat_compressed, Linear>;