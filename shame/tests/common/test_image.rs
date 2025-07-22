use std::fmt::Display;

use shame_wgpu as sm;
use wgpu::util::DeviceExt;

use crate::common::buffer_mapping::download_buffer_from_gpu;

pub trait TestImageFormat: sm::TextureFormat {
    type Texel: Copy + Default + AsRef<[u8]> + AsMut<[u8]> + PartialEq + Eq + bytemuck::NoUninit;
    const NUM_CHANNELS: usize;
}

#[rustfmt::skip] impl TestImageFormat for sm::tf::R8Uint         { type Texel = [u8; 1]; const NUM_CHANNELS: usize = 1; }
#[rustfmt::skip] impl TestImageFormat for sm::tf::Rg8Uint        { type Texel = [u8; 2]; const NUM_CHANNELS: usize = 2; }
#[rustfmt::skip] impl TestImageFormat for sm::tf::Rgba8Uint      { type Texel = [u8; 4]; const NUM_CHANNELS: usize = 4; }
#[rustfmt::skip] impl TestImageFormat for sm::tf::Rgba8Unorm     { type Texel = [u8; 4]; const NUM_CHANNELS: usize = 4; }
#[rustfmt::skip] impl TestImageFormat for sm::tf::Rgba8UnormSrgb { type Texel = [u8; 4]; const NUM_CHANNELS: usize = 4; }
#[rustfmt::skip] impl TestImageFormat for sm::tf::Rg8Unorm       { type Texel = [u8; 2]; const NUM_CHANNELS: usize = 2; }
#[rustfmt::skip] impl TestImageFormat for sm::tf::R8Unorm        { type Texel = [u8; 1]; const NUM_CHANNELS: usize = 1; }

/// a 2D image that can be created from - and displayed as - a monospaced 2D grid of
/// hex/unicode glyphs, arranged in columns for each color channel
/// ```
/// R                G
/// ░░░░░░░░FD░░░░░░ ░░░░░░░░CC░░░░░░
/// ░░░░░░░░FEFE░░░░ ░░░░░░░░FEFE░░░░
/// ░░░░░░░░▓▓▓▓▓▓░░ ░░░░░░░░▓▓▓▓▓▓░░
/// ░░░░░░░░▓▓▓▓▓▓▓▓ ░░░░░░░░▓▓▓▓▓▓▓▓
/// ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
/// ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
/// ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
/// ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
/// ```
/// each texel is represented by 2 characters.
/// special characters for improved readability are:
/// ```
/// ▓▓ = 0xFF
/// ▒▒ = 0x80
/// ░░ = 0x00
/// ```
/// all other values are displayed as their hex representation, e.g. the
/// top most texel of the shown triangle has color `{ red: 0xFD, green: 0xCC }`
#[derive(Clone, Copy, Eq)]
pub struct TestImage2D<Fmt: TestImageFormat, const W: usize, const H: usize> {
    data: [[Fmt::Texel; W]; H],
}

impl<Fmt: TestImageFormat, const W: usize, const H: usize> PartialEq for TestImage2D<Fmt, W, H> {
    fn eq(&self, other: &Self) -> bool { self.data == other.data }
}

impl<Fmt: TestImageFormat, const W: usize, const H: usize> std::fmt::Debug for TestImage2D<Fmt, W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self) }
}

impl<Fmt: TestImageFormat, const W: usize, const H: usize> Default for TestImage2D<Fmt, W, H> {
    fn default() -> Self {
        TestImage2D {
            data: [[Fmt::Texel::default(); W]; H],
        }
    }
}

impl<Fmt: TestImageFormat, const W: usize, const H: usize> Display for TestImage2D<Fmt, W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for channel_i in 0..Fmt::NUM_CHANNELS {
            f.write_str(match channel_i {
                0 => "R",
                1 => "G",
                2 => "B",
                3 => "A",
                _ => "?",
            })?;
            write!(f, " ")?;
            for _ in 1..W {
                write!(f, "  ")?;
            }
            write!(f, " ")?;
        }
        writeln!(f)?;
        for row in &self.data {
            for channel_i in 0..Fmt::NUM_CHANNELS {
                for texel in row {
                    match texel.as_ref().get(channel_i) {
                        Some(u8) => match u8 {
                            0xFF => write!(f, "▓▓")?,
                            0x80 => write!(f, "▒▒")?,
                            0x00 => write!(f, "░░")?,
                            _ => write!(f, "{:02x?}", u8)?,
                        },
                        None => write!(f, "??")?,
                    }
                }
                write!(f, " ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum TestImageError {
    InvalidString,
    InvalidChannelCount { expected: usize, got: usize },
}

impl Display for TestImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestImageError::InvalidString => write!(f, "invalid string"),
            TestImageError::InvalidChannelCount { expected, got } => {
                write!(f, "invalid channel count: expected {expected}, got {got}")
            }
        }
    }
}

impl<Fmt: TestImageFormat, const W: usize, const H: usize> TestImage2D<Fmt, W, H> {
    pub fn download_from_gpu(gpu: &sm::Gpu, texture: &wgpu::Texture) -> Self {
        // round up to next 256 multiple because of the `copy_texture_to_buffer` alignment requirement
        let bytes_per_row = size_of::<[Fmt::Texel; W]>().next_multiple_of(256) as u64;
        let byte_size = bytes_per_row * H as u64;

        let texture_size = wgpu::Extent3d {
            width: W as u32,
            height: H as u32,
            depth_or_array_layers: 1,
        };

        let staging_buffer = gpu.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TestImage2D download staging buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = gpu.create_command_encoder(&Default::default());

        enc.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row as u32),
                    rows_per_image: Some(H as u32),
                },
            },
            texture_size,
        );

        gpu.queue().submit([enc.finish()]);
        gpu.poll(wgpu::PollType::Wait).unwrap();

        let downloaded = download_buffer_from_gpu(&gpu, staging_buffer.slice(..)).unwrap();

        Self {
            data: {
                std::array::from_fn(|y| {
                    std::array::from_fn(|x| {
                        let mut texel = Fmt::Texel::default();
                        let size_of_texel = size_of::<Fmt::Texel>();
                        for (i, chan) in texel.as_mut().iter_mut().enumerate() {
                            *chan = downloaded[y * bytes_per_row as usize + x * size_of_texel + i]
                        }
                        texel
                    })
                })
            },
        }
    }

    pub fn upload_to_gpu(&self, gpu: &wgpu::Device, q: &wgpu::Queue) -> wgpu::Texture {
        let texture_size = wgpu::Extent3d {
            width: W as u32,
            height: H as u32,
            depth_or_array_layers: 1,
        };

        let texture = gpu.create_texture_with_data(
            q,
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: shame_wgpu::conversion::texture_format(&Fmt::id(), None).unwrap(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING |
                    wgpu::TextureUsages::RENDER_ATTACHMENT |
                    wgpu::TextureUsages::COPY_DST |
                    wgpu::TextureUsages::COPY_SRC,
                label: Some("test image"),
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(self.data.as_flattened()),
        );

        texture
    }
}

impl<Fmt: TestImageFormat, const W: usize, const H: usize> TestImage2D<Fmt, W, H> {
    pub fn try_from_str(str: &str) -> Result<Self, TestImageError> {
        let err = || TestImageError::InvalidString;
        let str = str.trim();

        let line0 = str.lines().next().ok_or_else(err)?;

        let words = line0.split_ascii_whitespace();
        if words.clone().count() > Fmt::NUM_CHANNELS {
            return Err(TestImageError::InvalidChannelCount {
                expected: Fmt::NUM_CHANNELS,
                got: words.clone().count(),
            });
        }

        let mut channel_order = {
            let mut order = Vec::new();
            for word in words {
                for (i, &ch) in ["R", "G", "B", "A"].iter().enumerate() {
                    if word == ch {
                        if order.contains(&i) {
                            return Err(err());
                        }
                        order.push(i);
                    }
                }
            }
            order
        };

        let mut lines = str.lines();
        match channel_order.is_empty() {
            true => {
                // no channel-line, start parsing content right away
                channel_order.push(0);
            }
            false => {
                // skip first line wrt content
                lines.next();
            }
        }

        // parse texels
        let mut data: [[Fmt::Texel; W]; H] = std::array::from_fn(|_| std::array::from_fn(|_| Default::default()));
        for (y, line) in lines.enumerate() {
            for (&channel_index, word) in channel_order.iter().zip(line.split_ascii_whitespace()) {
                let row: [u8; W] = parse_row(word).ok_or_else(err)?;
                for x in 0..row.len() {
                    let texel = &mut data[y][x].as_mut();
                    texel[channel_index] = row[x];
                }
            }
        }

        Ok(Self { data })
    }
}

#[allow(clippy::needless_range_loop)]
fn parse_row<const N: usize>(word: &str) -> Option<[u8; N]> {
    let mut arr = [0; N];
    if word.chars().count() != N * 2 {
        return None;
    }
    for i in 0..N {
        let texel_str = {
            let mut char_it = word
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(word.len()))
                .skip(i * 2);
            let start = char_it.next()?;
            char_it.next(); // every "texel" has 2 chars, skip middle boundary
            let end = char_it.next()?;
            &word[start..end]
        };

        arr[i] = match texel_str {
            "░░" => 0x00,
            "▒▒" => 0x80,
            "▓▓" => 0xFF,
            s if s.chars().count() == 2 => u8::from_str_radix(s, 16).ok()?,
            _ => unreachable!("texel_str should already be 2 chars guaranteed, but it is `{texel_str}`"),
        };
    }
    Some(arr)
}
