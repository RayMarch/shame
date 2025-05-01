use super::{mask::BitVec64, rasterizer::Winding};
use crate::{
    f32x1,
    frontend::{any::blend::Blend, error::InternalError, texture::texture_traits::Spp},
    ir::{recording::Context, TextureFormatWrapper},
    ToGpuType,
};

/// ## remove fragments based on their depth compared to the depth buffer
///
/// Depth testing is a common technique in realtime graphics, we recommend
/// searching the term online to find many illustrative examples of what it does.
///
/// This struct describes all possible depth test configurations.
///
/// Instead of filling out the constructor directly, there are also
/// shorthand initializer functions like
/// - `DepthTest::equal`
/// - `DepthTest::less`
/// - `DepthTest::less_equal`
/// - `DepthTest::greater`
/// - `DepthTest::greater_equal`
///
/// which provide the most common configurations.
#[derive(Debug, Clone, Copy)]
pub struct DepthTest {
    /// describes the comparison operator of the depth test, which is performed per fragment.
    ///
    /// use [`Test::LessEqual`] if unsure.
    ///
    /// The depth test comparison expression is defined as follows:
    ///
    /// [`operand`] [`test`] `depth-target texel`
    ///
    /// for example, if
    /// - [`operand`] is the triangle's depth at a fragment, e.g. `0.21`
    /// - [`test`] is [`Test::Less`], the `<` comparison operator
    /// - `depth-target texel` is the depth value at the fragment position in the depth buffer, e.g. `0.1`
    ///
    /// then the depth test calculates:
    ///
    /// `0.21 < 0.1`
    ///
    /// Which is false, so the depth test failed for this fragment, which
    /// indicates that the triangle is at least partially occluded by something
    /// that was rendered before into the same depth buffer.
    ///
    /// [`operand`]: DepthTest::operand
    /// [`test`]: DepthTest::test
    pub test: Test,
    /// the left-hand-side operand in the depth-test comparison operation, defined by [`DepthTest::test`]
    ///
    /// use [`DepthLhs::default()`] if unsure.
    pub operand: DepthLhs,
    /// whether to replace the depth-target's texel value with the [`DepthTest::operand`]
    /// value if the test passes.
    ///
    /// use `true` if unsure.
    ///
    /// This flag is often referred to as "z-write" or "depth mask".
    pub replace_on_pass: bool,
}

impl Default for DepthTest {
    fn default() -> Self {
        Self {
            test: Test::LessEqual,
            operand: DepthLhs::FragmentZ(DepthBias::default()),
            replace_on_pass: true,
        }
    }
}

impl DepthTest {
    /// creates a `DepthTest` which compares the rasterizer per-fragment z coordinate
    /// (derived from the rasterizer clip-space position)
    /// to the depth-target's texel depth value.
    pub fn vs_fragment_z(test: Test, replace_on_pass: bool) -> Self {
        DepthTest {
            test,
            operand: DepthLhs::FragmentZ(DepthBias::default()),
            replace_on_pass,
        }
    }

    /// only keep fragments with a depth that is less or equal to the value
    /// stored in the depth buffer at the fragments xy position.
    ///
    /// ### `replace_on_pass`:
    /// whether to replace the depth-target's pixel value with the `operand` value if the test passes.
    /// This flag is often referred to as "z-write" or "depth mask".
    ///
    /// use `true` if unsure.
    pub fn less_equal(replace_on_pass: bool) -> Self { DepthTest::vs_fragment_z(Test::LessEqual, replace_on_pass) }

    /// only keep fragments with a depth that is less than the value
    /// stored in the depth buffer at the fragments xy position.
    ///
    /// ### `replace_on_pass`:
    /// whether to replace the depth-target's pixel value with the fragment's value if the test passes.
    /// This flag is often referred to as "z-write" or "depth mask".
    ///
    /// use `true` if unsure.
    pub fn less(replace_on_pass: bool) -> Self { DepthTest::vs_fragment_z(Test::Less, replace_on_pass) }

    /// only keep fragments with a depth that is greater than the value
    /// stored in the depth buffer at the fragments xy position.
    ///
    /// ### `replace_on_pass`:
    /// whether to replace the depth-target's pixel value with the fragment's value if the test passes.
    /// This flag is often referred to as "z-write" or "depth mask".
    ///
    /// use `true` if unsure.
    pub fn greater(replace_on_pass: bool) -> Self { DepthTest::vs_fragment_z(Test::Greater, replace_on_pass) }

    /// only keep fragments with a depth that is greater than or equal to the value
    /// stored in the depth buffer at the fragments xy position.
    ///
    /// ### `replace_on_pass`:
    /// whether to replace the depth-target's pixel value with the fragment's value if the test passes.
    /// This flag is often referred to as "z-write" or "depth mask".
    ///
    /// use `true` if unsure.
    pub fn greater_equal(replace_on_pass: bool) -> Self {
        DepthTest::vs_fragment_z(Test::GreaterEqual, replace_on_pass)
    }

    /// keep all fragments regardless of the value
    /// stored in the depth buffer at the fragments xy position.
    ///
    /// ### `replace`:
    /// whether to replace the depth-target's pixel value with the fragment's value.
    /// This flag is often referred to as "z-write" or "depth mask".
    ///
    /// use `true` if unsure.
    pub fn always_pass(replace: bool) -> Self { DepthTest::vs_fragment_z(Test::Always, replace) }

    /// remove all fragments regardless of the depth values
    /// stored in the depth buffer
    pub fn always_fail() -> Self { DepthTest::vs_fragment_z(Test::Never, false) }

    /// only keep fragments with a depth that is not equal to the value
    /// stored in the depth buffer at the fragments xy position.
    ///
    /// ### `replace_on_pass`:
    /// whether to replace the depth-target's pixel value with the fragment's value if the test passes.
    /// This flag is often referred to as "z-write" or "depth mask".
    ///
    /// use `true` if unsure.
    pub fn not_equal(replace_on_pass: bool) -> Self { DepthTest::vs_fragment_z(Test::NotEqual, replace_on_pass) }

    /// only keep fragments with a depth that is equal to the value
    /// stored in the depth buffer at the fragments xy position.
    ///
    /// ### `replace_on_pass`:
    /// whether to replace the depth-target's pixel value with the fragment's value if the test passes.
    /// This flag is often referred to as "z-write" or "depth mask".
    ///
    /// use `true` if unsure.
    pub fn equal(replace_on_pass: bool) -> Self { DepthTest::vs_fragment_z(Test::Equal, replace_on_pass) }

    /// keep all fragments and replace the depth stored in the depth buffer
    /// with the value of those fragments in the respective xy position
    ///
    /// this is equivalent to
    /// ```
    /// DepthTest::always_pass(true)
    /// ```
    pub fn replace() -> Self { DepthTest::always_pass(true) }
}

/// the left hand side operand of the depth-test operation.
///
/// If [`DepthTest::replace_on_pass`] is enabled, this value will also be used to
/// fill the depth buffer with per-fragment values, if the depth test succeeds for
/// any given fragment.
///
/// [`DepthTest::replace_on_pass`]: crate::DepthTest::replace_on_pass
#[derive(Clone, Copy)]
pub enum DepthLhs {
    /// A per-fragment [`f32x1`] value
    /// (this value is then converted to the [`DepthFormat`] of the respective depth buffer)
    ///
    /// [`DepthFormat`]: crate::DepthFormat
    Explicit(f32x1, DepthBias),
    /// use the z-value that was interpolated by the rasterizer for every fragment
    /// of the primitive
    FragmentZ(DepthBias),
}

impl From<f32> for DepthLhs {
    fn from(value: f32) -> Self { DepthLhs::Explicit(value.to_gpu(), DepthBias::default()) }
}

impl From<f32x1> for DepthLhs {
    fn from(value: f32x1) -> Self { DepthLhs::Explicit(value, DepthBias::default()) }
}

impl std::fmt::Debug for DepthLhs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Explicit(_, arg1) => f.debug_tuple("Explicit").field(arg1).finish(),
            Self::FragmentZ(arg0) => f.debug_tuple("FragmentZ").field(arg0).finish(),
        }
    }
}

impl Default for DepthLhs {
    fn default() -> Self { DepthLhs::FragmentZ(DepthBias::default()) }
}

/// (no documentation yet)
#[derive(Debug, Clone, Copy)]
pub enum StencilBranch {
    // regular stencil test with separate [`StencilOp`]s for passing/failing the [`Test`]
    /// (no documentation yet)
    Test {
        // comparison: "`stencil_ref` `test` `stencil_target`", where `stencil_ref` is the stencil reference value provided by the graphics api.
        /// (no documentation yet)
        test: Test,
        /// (no documentation yet)
        on_pass: StencilOp,
        /// (no documentation yet)
        on_fail: StencilOp,
    },
    /// stencil test with separate [`StencilOp`]s for
    /// - passing the stencil [`Test`] and passing the [`DepthTest`]
    /// - passing the stencil [`Test`] and failing the [`DepthTest`]
    /// - failing the stencil [`Test`]
    ///
    /// see:
    /// - webgpu: https://gpuweb.github.io/gpuweb/#output-merging
    /// - vulkan: https://docs.vulkan.org/spec/latest/chapters/fragops.html#fragops-stencil
    TestConsiderDepth {
        /// comparison: "`stencil_ref` `test` `stencil_target`", where `stencil_ref` is the stencil reference value provided by the graphics api.
        test: Test,
        /// (no documentation yet)
        on_pass_depth_pass: StencilOp,
        /// (no documentation yet)
        on_pass_depth_fail: StencilOp,
        /// (no documentation yet)
        on_fail: StencilOp,
    },
}

/// (no documentation yet)
#[derive(Debug, Clone, Copy)]
pub enum StencilTest {
    /// single stencil test defined for for the specified winding order(s)
    Single(Winding, StencilMasking, StencilBranch),
    /// separate stencil tests for primitives with clockwise and
    /// counterclockwise winding
    PerWinding {
        /// (no documentation yet)
        masking: StencilMasking,
        /// (no documentation yet)
        ccw: StencilBranch,
        /// (no documentation yet)
        cw: StencilBranch,
    },
}

/// (no documentation yet)
#[derive(Default, Debug, Clone, Copy)]
pub enum StencilMasking {
    /// (no documentation yet)
    #[default]
    Unmasked,
    /// (no documentation yet)
    Masked(u8),
    /// (no documentation yet)
    PerAccess {
        /// Stencil values are AND'd with this mask when reading and writing the stencil buffer
        read_write: u8,
        /// Stencil values are AND'd with this mask when writing the stencil buffer
        write: u8,
    },
}

/// (no documentation yet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[rustfmt::skip]
pub struct StencilState {
    /// the stencil test applied to counter-clockwise faces.
    pub ccw: StencilFace,
    /// the stencil test applied to clockwise faces.
    pub cw: StencilFace, 
    /// Stencil values are AND'd with this mask when reading and writing the stencil buffer. Only low 8 bits are used.
    pub rw_mask: u32, 
    /// Stencil values are AND'd with this mask when writing the stencil buffer. Only low 8 bits are used. 
    pub w_mask: u32, 
}

/// see https://www.w3.org/TR/webgpu/#dictdef-gpustencilfacestate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFace {
    /// comparison operation that determines whether the stencil test passes or fails.
    ///
    /// see https://www.w3.org/TR/webgpu/#dictdef-gpustencilfacestate
    pub compare: Test,
    /// op when depth test fails but stencil test succeeds.
    pub on_pass_depth_fail: StencilOp,
    /// op when stencil test fails.
    pub on_fail: StencilOp,
    /// op when both depth and stencil test succeed.
    pub on_pass_depth_pass: StencilOp,
}

/// A comparison operation
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Test {
    /// never passes
    Never,
    Less,
    // TODO(release) represent in shame: make `Equal` force `Accuracy::Reproducible` to prevent artifacts?
    Equal,
    LessEqual,
    Greater,
    // TODO(release) represent in shame: make `Equal` force `Accuracy::Reproducible` to prevent artifacts?
    NotEqual,
    GreaterEqual,
    // always passes
    Always,
}

/// the operation applied to the stencil buffer
///
/// see https://www.w3.org/TR/webgpu/#enumdef-gpustenciloperation
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StencilOp {
    /// Keep the current stencil value
    #[default]
    Keep,
    /// Set the stencil value to 0
    Zero,
    /// Set the stencil value to the value of the stencil reference
    Replace,
    /// Bitwise-invert the current stencil value
    Invert,
    /// Increments the current stencil value, clamping to the maximum
    /// representable value of the depth/stencil attachment's stencil aspect
    IncClamp,
    /// Decrement the current stencil value, clamping to 0
    DecClamp,
    /// Increments the current stencil value, wrapping to zero if the value
    /// exceeds the maximum representable value of the depth/stencil attachment's
    /// stencil aspect
    IncWrap,
    /// Decrement the current stencil value, wrapping to the maximum
    /// representable value of the depthStencilAttachment’s stencil
    /// aspect if the value goes below 0
    DecWrap,
}

/// information about the depth and stencil test, as well as target formats
///
/// see https://www.w3.org/TR/webgpu/#dictdef-gpudepthstencilstate
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DepthStencilState {
    /// (no documentation yet)
    pub format: TextureFormatWrapper,
    /// (no documentation yet)
    pub depth_write_enabled: bool,
    /// (no documentation yet)
    pub depth_compare: Test,
    /// (no documentation yet)
    pub bias: DepthBias,
    /// (no documentation yet)
    pub stencil: StencilState,
}

/// (no documentation yet)
/// if unsure use `DepthBias::default()`
// TODO(release) must be `DepthBias::zero()` for non-triangle `shame::Draw`, enforce this
#[derive(Debug, Copy, Clone)]
pub struct DepthBias {
    // Constant depth biasing factor, in basic units of the depth format.
    /// (no documentation yet)
    pub constant: i32,
    // Slope depth biasing factor.
    /// (no documentation yet)
    pub slope_scale: f32,
    // Depth bias clamp value (absolute).
    /// (no documentation yet)
    pub clamp: f32,
}

impl DepthBias {
    /// no depth bias
    pub fn zero() -> Self {
        Self {
            constant: 0,
            slope_scale: 0.0,
            clamp: 0.0,
        }
    }
}

impl std::hash::Hash for DepthBias {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.constant.hash(state);
        // this may not be strictly correct
        self.slope_scale.to_bits().hash(state);
        self.clamp.to_bits().hash(state);
    }
}

impl PartialEq for DepthBias {
    fn eq(&self, other: &Self) -> bool {
        self.constant == other.constant &&
        // treat NaNs equally
        self.slope_scale.to_bits() == other.slope_scale.to_bits() &&
        self.clamp.to_bits() == other.clamp.to_bits()
    }
}
impl Eq for DepthBias {}

impl Default for DepthBias {
    fn default() -> Self { DepthBias::zero() }
}

impl From<StencilBranch> for StencilFace {
    fn from(value: StencilBranch) -> Self {
        match value {
            StencilBranch::Test { test, on_pass, on_fail } => StencilFace {
                compare: test,
                on_pass_depth_fail: on_pass,
                on_pass_depth_pass: on_pass,
                on_fail,
            },
            StencilBranch::TestConsiderDepth {
                test,
                on_pass_depth_pass,
                on_pass_depth_fail,
                on_fail,
            } => StencilFace {
                compare: test,
                on_pass_depth_fail,
                on_pass_depth_pass,
                on_fail,
            },
        }
    }
}

impl StencilTest {
    pub(crate) fn to_stencil_state(
        self,
        ccw_is_front: bool,
        num_stencil_bits: u32,
    ) -> Result<StencilState, InternalError> {
        let unmask: u8 = 0b11111111;
        let ignore_branch = StencilBranch::Test {
            test: Test::Never,
            on_pass: StencilOp::Keep,
            on_fail: StencilOp::Keep,
        };
        let masking: StencilMasking;
        let branch_ccw;
        let branch_cw;

        match self {
            StencilTest::Single(w, m, b) => {
                masking = m;
                match w {
                    Winding::Ccw => {
                        branch_ccw = b;
                        branch_cw = ignore_branch;
                    }
                    Winding::Cw => {
                        branch_ccw = ignore_branch;
                        branch_cw = b;
                    }
                    Winding::Either => {
                        branch_ccw = b;
                        branch_cw = b;
                    }
                }
            }
            StencilTest::PerWinding { masking: m, ccw, cw } => {
                masking = m;
                branch_ccw = ccw;
                branch_cw = cw;
            }
        }

        let amount_of_bits = |mask: u8| {
            // this exists because i was switching bitmask type implementations a few times,
            // and will probably do so again in the future.
            // Relying on something like `std::mem::size_of_val(x) * 8` would not cause
            // a compiler error if we switch to a type with runtime-defined amount of bits
            // (e.g. BitVec64 or any other bit-vec). Instead it would silently
            // produce a bug.
            mask.count_zeros() + mask.count_ones()
        };
        let (rw_mask, w_mask) = match masking {
            StencilMasking::Unmasked => (unmask, unmask),
            StencilMasking::Masked(mask) => {
                let num_mask_bits = amount_of_bits(mask);
                if num_mask_bits < num_stencil_bits {
                    return Err(InternalError::new(
                        true,
                        format!(
                            "stencil mask ({num_mask_bits} bits) not big enough for target ({num_stencil_bits} bits)"
                        ),
                    ));
                }
                (mask, mask)
            }
            StencilMasking::PerAccess { read_write, write } => {
                let num_rw_bits = amount_of_bits(read_write);
                if num_rw_bits < num_stencil_bits {
                    return Err(InternalError::new(
                        true,
                        format!(
                            "stencil read_write mask ({num_rw_bits} bits) not big enough for target ({num_stencil_bits} bits)"
                        ),
                    ));
                }
                let num_w_bits = amount_of_bits(write);
                if num_w_bits < num_stencil_bits {
                    return Err(InternalError::new(
                        true,
                        format!(
                            "stencil write mask ({num_w_bits} bits) not big enough for target ({num_stencil_bits} bits)"
                        ),
                    ));
                }
                (read_write, write)
            }
        };

        Ok(StencilState {
            ccw: if ccw_is_front { branch_ccw } else { branch_cw }.into(),
            cw: if ccw_is_front { branch_cw } else { branch_ccw }.into(),
            rw_mask: rw_mask as u32,
            w_mask: w_mask as u32,
        })
    }
}
