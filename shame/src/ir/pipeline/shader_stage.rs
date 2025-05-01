use std::{
    fmt::{Display, Write},
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Neg, Not},
};

use super::PipelineKind;

/// makes rustc check that all variants are mentioned exhaustively.
/// triggers a compiler error if not.
macro_rules! exhaustive_all_variants {
    (
        [
            $($variant: path),* $(,)?
        ]
    ) => {
        if (false) {
            match unreachable!() {
                $($variant => ()),*
            };
            unreachable!()
        } else {
            [
                $($variant),*
            ]
        }
    };
}

/// A pipeline shader stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShaderStage {
    /// compute shader stage. The only stage of a compute pipeline
    Comp,
    /// task shader stage. Part of the mesh pipeline
    Task,
    /// mesh shader stage. Part of the mesh pipeline
    Mesh,
    /// vertex shader stage. Part of the render pipeline
    Vert,
    /// fragment shader stage. Part of the render pipeline
    Frag,
}

impl ShaderStage {
    /// an array of all possible shader stages
    pub fn all() -> [ShaderStage; 5] {
        exhaustive_all_variants!([
            ShaderStage::Comp,
            ShaderStage::Task,
            ShaderStage::Mesh,
            ShaderStage::Vert,
            ShaderStage::Frag,
        ])
    }

    /// array containing all shader stages where, if a stage happens before another
    /// stage, that stage is earlier in the list.
    pub fn order() -> [ShaderStage; 5] {
        exhaustive_all_variants!([
            ShaderStage::Comp,
            ShaderStage::Task,
            ShaderStage::Mesh,
            ShaderStage::Vert,
            ShaderStage::Frag,
        ])
    }
}

impl std::fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ShaderStage::Comp => "compute",
            ShaderStage::Task => "task",
            ShaderStage::Mesh => "mesh",
            ShaderStage::Vert => "vertex",
            ShaderStage::Frag => "fragment",
        })
    }
}

impl ShaderStage {
    /// a `StageMask` with only the bit of `self` set.
    pub fn as_mask(&self) -> StageMask { From::from(*self) }
}

/// A bitmask where every shader stage has a unique bit position.
///
/// This mask can hold bits of shaderstages from different pipeline kinds simultaneously.
/// For example, it can hold a compute-shader bit and a vertex-shader bit
/// at the same time, even though these shaders can never coexist in a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct StageMask(u8);

impl StageMask {
    /// all possible shader stages of any pipeline kind
    pub const fn all() -> StageMask { StageMask(0b11111) }
    /// none of the shader stages
    pub const fn empty() -> StageMask { StageMask(0b00000) }
    /// the compute shader stage, used in compute pipelines
    pub const fn comp() -> StageMask { StageMask(0b00001) }
    /// the task shader stage, used in mesh pipelines
    pub const fn task() -> StageMask { StageMask(0b00010) }
    /// the mesh shader stage, used in mesh pipelines
    pub const fn mesh() -> StageMask { StageMask(0b00100) }
    /// the vertex shader stage, used in render pipelines
    pub const fn vert() -> StageMask { StageMask(0b01000) }
    /// the fragment shader stage, used in render pipelines and mesh pipelines
    pub const fn frag() -> StageMask { StageMask(0b10000) }
    /// all stages of a mesh pipeline (task, mesh, fragment)
    pub const fn pipeline_mesh() -> StageMask { Self::mesh().or(Self::task()).or(Self::frag()) }
    /// all stages of a render pipeline (vertex, fragment)
    pub const fn pipeline_render() -> StageMask { Self::vert().or(Self::frag()) }
    /// all stages of a compute pipeline (compute shader only)
    pub const fn pipeline_compute() -> StageMask { Self::comp() }
    /// all stages of a given pipeline kind
    pub fn pipeline(pipeline: PipelineKind) -> StageMask {
        match pipeline {
            PipelineKind::Render => Self::pipeline_render(),
            PipelineKind::Compute => Self::pipeline_compute(),
        }
    }

    const fn or(self, other: Self) -> Self { Self(self.0 | other.0) }

    /// whether `self` contains no shader stages at all
    pub fn is_empty(self) -> bool { self.0 == 0 }

    /// the amount of shader stages in `self`
    pub fn count_ones(self) -> u32 { self.0.count_ones() }

    /// returns the stage `Some(stage)` if `self` only contains a single shader stage
    pub fn get_only_stage(self) -> Option<ShaderStage> {
        if self.0.count_ones() == 1 {
            if let Some(s) = self.into_iter().next() {
                return Some(s);
            }
        }
        None
    }

    /// returns a stage mask that only contains the earliest stage of `self` wrt their order in pipelines.
    /// Empty input masks are returned unmodified
    ///
    /// e.g. turns vert|frag into vert
    pub fn earliest_stage_only(mut self) -> Self {
        for stage in ShaderStage::order() {
            if self.contains_stage(stage) {
                return stage.into();
            }
        }
        self
    }

    #[doc(hidden)] // internal
    #[allow(clippy::wrong_self_convention)]
    pub fn to_string_verbose(&self) -> String {
        let mut s = String::new();
        let mut written_so_far = 0;
        s.write_str("[");
        for stage in ShaderStage::all() {
            if self.contains_stage(stage) {
                if written_so_far != 0 {
                    write!(s, ", ");
                }
                write!(s, "{stage}");
                written_so_far += 1;
            }
        }
        s.write_str("]");
        s
    }

    pub(crate) fn contains_stage(self, stage: ShaderStage) -> bool {
        let stage = StageMask::from(stage);
        (self & stage) != StageMask::empty()
    }
}

/// iterator over all stages in a stage mask
pub struct StageMaskIter(StageMask);

impl Iterator for StageMaskIter {
    type Item = ShaderStage;

    fn next(&mut self) -> Option<Self::Item> {
        ShaderStage::all()
            .into_iter()
            .find(|stage| self.0.contains_stage(*stage))
            .inspect(|stage| self.0 &= !StageMask::from(*stage))
    }
}

impl IntoIterator for StageMask {
    type Item = ShaderStage;

    type IntoIter = StageMaskIter;

    fn into_iter(self) -> Self::IntoIter { StageMaskIter(self) }
}

impl Display for StageMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return f.write_char('-');
        }

        for (stage, name) in [
            (StageMask::frag(), 'f'),
            (StageMask::vert(), 'v'),
            (StageMask::task(), 't'),
            (StageMask::mesh(), 'm'),
            (StageMask::comp(), 'c'),
        ] {
            if stage & *self != StageMask::empty() {
                f.write_char(name)?
            }
        }
        Ok(())
    }
}

impl From<ShaderStage> for StageMask {
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Comp => StageMask::comp(),
            ShaderStage::Task => StageMask::task(),
            ShaderStage::Mesh => StageMask::mesh(),
            ShaderStage::Vert => StageMask::vert(),
            ShaderStage::Frag => StageMask::frag(),
        }
    }
}

impl From<PipelineKind> for StageMask {
    fn from(value: PipelineKind) -> Self {
        match value {
            PipelineKind::Render => StageMask::pipeline_render(),
            PipelineKind::Compute => StageMask::pipeline_compute(),
        }
    }
}

impl Not for StageMask {
    type Output = Self;

    fn not(self) -> Self::Output { Self(self.0.not()) }
}

impl BitAnd for StageMask {
    type Output = StageMask;
    fn bitand(self, rhs: Self) -> Self::Output { StageMask(self.0 & rhs.0) }
}

impl BitAndAssign for StageMask {
    fn bitand_assign(&mut self, rhs: Self) { *self = *self & rhs }
}

impl BitOr for StageMask {
    type Output = StageMask;
    fn bitor(self, rhs: Self) -> Self::Output { StageMask(self.0 | rhs.0) }
}

impl BitOrAssign for StageMask {
    fn bitor_assign(&mut self, rhs: Self) { *self = *self | rhs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_order() {
        assert!(StageMask::vert() < StageMask::frag());
        assert!(StageMask::task() < StageMask::mesh());
        assert!(StageMask::mesh() < StageMask::frag());

        assert!(StageMask::empty() < StageMask::vert());
        assert!(StageMask::empty() < StageMask::task());
        assert!(StageMask::frag() < StageMask::all());

        assert!(ShaderStage::Vert < ShaderStage::Frag);
        assert!(ShaderStage::Task < ShaderStage::Mesh);
        assert!(ShaderStage::Mesh < ShaderStage::Frag);
    }
}
