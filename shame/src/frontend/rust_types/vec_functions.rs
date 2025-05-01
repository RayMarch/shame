#![allow(clippy::identity_op)]
use std::cmp::Ordering;

use super::{
    len::{AtLeastLen, Len, Len2},
    scalar_type::{ScalarType, ScalarType32Bit, ScalarTypeFp, ScalarTypeNumber},
    type_traits::NoBools,
    vec::{scalar, IsVec, ToInteger, ToScalar, ToVec},
    AsAny, GpuType, To, ToGpuType,
};
use crate::{common::floating_point::f16, frontend::encoding::flow::discard_if, ScalarTypeInteger, ScalarTypeSigned};
use crate::frontend::rust_types::len::{x1, x2, x3, x4};
use crate::frontend::rust_types::vec::vec;
use crate::{
    call_info,
    frontend::{
        any::{flow_builders::IfRecorder, Any, InvalidReason},
        encoding::flow::FlowFn,
        rust_types::{
            vec_range::{VecBounds, VecBoundsByLen},
            vec_range_traits::{VecRange, VecRangeBounds, VecRangeInclusive, VecRangeBoundsInclusive},
        },
    },
    ir::{
        self,
        recording::{Context, FrexpGenerics, ModfGenerics},
        ScalarConstant,
    },
};

impl<T: ScalarTypeFp> vec<T, x1> {
    /// Returns the component-wise smooth Hermite interpolation between 0 and 1.
    ///
    /// ## Example
    /// ```
    /// let color = shame::vec!(1.0, 1.0, 1.0, 0.5);
    /// let alpha = color.w;
    /// let alpha = alpha.smoothstep(0.3..0.7);
    /// ```
    ///
    /// an alternative [`vec::smoothstep_each`] function exists for non-scalars
    ///
    /// for a given `range` of `from..to` the result is
    /// ```
    /// let t = ((self - from) / (to - from)).clamp(0.0 ..= 1.0);
    /// t * t * (3.0 - 2.0 * t)
    /// ```
    ///
    /// Qualitatively:
    ///
    /// for a given `range` of `from..to`:
    /// * When `from` < `to`,
    ///   the function is 0 for x below `from`, then smoothly rises until x reaches `to`, and remains at 1 afterward.
    /// * When `from` > `to`,
    ///   the function is 1 for x below `to`, then smoothly descends until x reaches `from`, and remains at 0 afterward.
    ///
    /// see <https://www.w3.org/TR/WGSL/#smoothstep-builtin>
    pub fn smoothstep(self, range: impl VecRange<T, x1>) -> vec<T, x1> {
        let [(from, _), (to, _)] = range.get_bounds().scalar();
        self.smoothstep_each_impl(from, to).as_any().into()

        // match range.get_bounds().by_len::<T, x1>() {
        //     VecBoundsByLen::X1([(from, _), (to, _)]) => self.smoothstep_each_impl(from, to).as_any().into(),
        //     VecBoundsByLen::L([(from, _), (to, _)]) => self.splat::<L>().smoothstep_each_impl(from, to),
        // }
    }

    /// Returns the linear blend of `from` and `to`
    /// where `self` is the blend factor
    /// ```
    /// from * (vec::one() - self) + to * self
    /// ```
    /// this function is also known as `mix` (see <https://www.w3.org/TR/WGSL/#mix-builtin>).
    ///
    /// an alternative component wise [`vec::lerp_each`] function exists for non-scalar blend factors
    ///
    pub fn lerp<L: Len>(self, from: impl To<vec<T, L>>, to: impl To<vec<T, L>>) -> vec<T, L> {
        // TODO(low prio) use `ScalarType`::LerpOutput to enable more scalar types than just Floating Point
        self.splat::<L>().lerp_each(from, to)
    }

    /// returns `vec!(self.cos(), self.sin())`
    #[track_caller]
    pub fn cos_sin(self) -> vec<T, x2>
    where
        T: ScalarTypeFp,
    {
        (self.cos(), self.sin()).to_gpu()
    }
}

impl<T: ScalarType, L: Len> vec<T, L> {
    /// component wise sin
    ///
    /// Returns the sine of `self`, where `self` is in radians.
    ///
    /// see <https://www.w3.org/TR/WGSL/#sin-builtin>
    #[track_caller]
    pub fn sin(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().sin().into()
    }

    /// component wise cos
    ///
    /// Returns the cosine of `self`, where `self` is in radians.
    ///
    /// see <https://www.w3.org/TR/WGSL/#cos-builtin>
    #[track_caller]
    pub fn cos(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().cos().into()
    }

    /// component wise sinh
    ///
    /// Returns the hyperbolic sine of `self`, where `self` is a hyperbolic angle.
    ///
    /// see <https://www.w3.org/TR/WGSL/#sinh-builtin>
    #[track_caller]
    pub fn sinh(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().sinh().into()
    }

    /// component wise cosh
    ///
    /// Returns the hyperbolic cosine of `self`, where `self` is a hyperbolic angle.
    ///
    /// see <https://www.w3.org/TR/WGSL/#cosh-builtin>
    #[track_caller]
    pub fn cosh(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().cosh().into()
    }

    /// component wise asin
    ///
    /// Returns the principal value, in radians, of the inverse sine (sin^-1) of `self`.
    ///
    /// see <https://www.w3.org/TR/WGSL/#asin-builtin>
    #[track_caller]
    pub fn asin(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().asin().into()
    }

    /// component wise acos
    ///
    /// Returns the principal value, in radians, of the inverse cosine (cos^-1) of `self`.
    ///
    /// see <https://www.w3.org/TR/WGSL/#acos-builtin>
    #[track_caller]
    pub fn acos(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().acos().into()
    }

    /// component wise asinh
    ///
    /// Returns the inverse hyperbolic sine (sinh^-1) of `self`, as a hyperbolic angle.
    ///
    /// see <https://www.w3.org/TR/WGSL/#asinh-builtin>
    #[track_caller]
    pub fn asinh(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().asinh().into()
    }

    /// component wise acosh
    ///
    /// Returns the inverse hyperbolic cosine (cosh^-1) of `self`, as a hyperbolic angle.
    ///
    /// see <https://www.w3.org/TR/WGSL/#acosh-builtin>
    #[track_caller]
    pub fn acosh(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().acosh().into()
    }

    /// component wise tan
    ///
    /// Returns the tangent of `self`, where `self` is in radians
    ///
    /// see <https://www.w3.org/TR/WGSL/#tan-builtin>
    #[track_caller]
    pub fn tan(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().tan().into()
    }

    /// component wise tanh
    ///
    /// Returns the hyperbolic tangent of `self`, where `self` is a hyperbolic angle.
    ///
    /// see <https://www.w3.org/TR/WGSL/#tanh-builtin>
    #[track_caller]
    pub fn tanh(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().tanh().into()
    }

    /// component wise atan
    ///
    /// Returns the principal value, in radians, of the inverse tangent (tan^-1) of `self`.
    /// That is, approximates x with − π/2 ≤ x ≤ π/2, such that tan(x) = `self`.
    ///
    /// see <https://www.w3.org/TR/WGSL/#atan-builtin>
    #[track_caller]
    pub fn atan(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().atan().into()
    }

    /// component wise atanh
    ///
    /// Returns the inverse hyperbolic tangent (tanh^-1) of `self`, as a hyperbolic angle.
    /// That is, approximates a such that tanh(a) = `self`.
    ///
    /// see <https://www.w3.org/TR/WGSL/#atanh-builtin>
    #[track_caller]
    pub fn atanh(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().atanh().into()
    }

    /// component wise `atan2`
    ///
    /// Returns an angle, in radians, in the interval [-π, π] whose tangent is y÷x.
    /// The quadrant selected by the result depends on the signs of y and x.
    ///
    /// see <https://www.w3.org/TR/WGSL/#atan2-builtin>
    #[track_caller]
    pub fn atan2_each(y: vec<T, L>, x: vec<T, L>) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        Any::atan2(y.as_any(), x.as_any()).into()
    }

    /// Converts radians to degrees (component wise).
    ///
    /// multiplies `self` with 360 / 2π
    #[track_caller]
    pub fn to_degrees(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().degrees().into()
    }

    /// Converts radians to degrees (component wise).
    ///
    /// multiplies `self` with 2π / 360
    #[track_caller]
    pub fn to_radians(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().radians().into()
    }

    /// component wise natural exponentiation of `self`
    ///
    /// returns e^`self`
    ///
    /// (e raised to the power of `self`)
    ///
    /// see <https://www.w3.org/TR/WGSL/#exp-builtin>
    #[track_caller]
    pub fn exp(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().exp().into()
    }

    /// component wise 2^`self`
    ///
    /// (2 raised to the power of `self`)
    ///
    /// see <https://www.w3.org/TR/WGSL/#exp2-builtin>
    #[track_caller]
    pub fn exp2(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().exp2().into()
    }

    /// component wise natural logarithm of `self`
    ///
    /// see <https://www.w3.org/TR/WGSL/#log-builtin>
    #[track_caller]
    pub fn ln(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().log().into()
    }

    /// component wise base 2 logarithm of `self`
    ///
    /// see <https://www.w3.org/TR/WGSL/#log2-builtin>
    #[track_caller]
    pub fn log2(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().log2().into()
    }

    /// combine fractional `self` and exponent `exp` part, component wise
    /// (this function is also known as `ldexp`)
    ///
    /// the relationship between `as_fract_with_exp` and `fract_exp` is:
    /// ```
    /// let (fract, exp) = x.fract_exp();
    /// x = fract.as_fract_with_exp(exp);
    /// ```
    ///
    /// see <https://www.w3.org/TR/WGSL/#ldexp-builtin>
    #[track_caller]
    pub fn as_fract_with_exp<I>(self, exp: vec<I, L>) -> vec<T, L>
    where
        I: ScalarTypeInteger,
        T: ScalarTypeFp,
    {
        Any::ldexp(self.as_any(), exp.as_any()).into()
    }

    /// floating point exponentiation
    ///
    /// returns each component of `self` raised to the power of `exponent`
    ///
    /// see <https://www.w3.org/TR/WGSL/#pow-builtin>
    #[track_caller]
    pub fn powf(self, exponent: impl To<vec<T, x1>>) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.powf_each(exponent.splat())
    }

    /// component wise integer exponentiation
    ///
    /// multiplies `self` with itself `exponent` times
    #[track_caller]
    pub fn powi_const(self, exponent: u32) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        match exponent {
            0 => vec::one(),
            1.. => {
                let mut val = self;
                for _ in 1..exponent {
                    val = val.mul_each(self);
                }
                val
            }
        }
    }

    /// component wise float exponentiation
    ///
    /// raises every component of `self` to the power of the corresponding component in `exponents`
    /// and returns the results as a vector.
    ///
    /// see <https://www.w3.org/TR/WGSL/#pow-builtin>
    #[track_caller]
    pub fn powf_each(self, exponents: vec<T, L>) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().pow(exponents.as_any()).into()
    }

    /// square root of `self`
    ///
    /// see <https://www.w3.org/TR/WGSL/#sqrt-builtin>
    #[track_caller]
    pub fn sqrt(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().sqrt().into()
    }

    #[allow(clippy::assign_op_pattern)]
    fn remap_literals(&self, [a0, a1]: [f64; 2], [b0, b1]: [f64; 2], ty: ir::ScalarType) -> Self {
        let diff_a = a1 - a0;
        let diff_b = b1 - b0;
        let fac = diff_b / diff_a;


        let fac = ((b1 - b0) / (a1 - a0));

        let as_any = |x| Any::new_scalar(ty.constant_from_f64(x));

        // = (self - a0) * (diff_b / diff_a) + b0
        // = (self - a0) * fac + b0;
        // = self * fac - a0 * fac + b0
        // = self * fac - (a0 * fac) + b0

        let mut r = self.as_any();
        // * fac
        if (fac != 1.0) {
            r = r * as_any(fac)
        }
        //  - (a0 * fac) + b0
        let offset = -(a0 * fac) + b0;
        match (offset).total_cmp(&0.0) {
            Ordering::Greater => r = r + as_any(offset),
            Ordering::Equal => (),
            Ordering::Less => r = r - as_any(-offset),
        }
        r.into()
    }

    /// linear remapping from the `from` range to the `to` range.
    ///
    /// no clamping happens, so values outside of the `from` range get remapped as well
    /// according to the same transformation.
    ///
    /// ## example
    /// ```
    /// let normal = sm::vec!(1.0, 0.3, -1.0).normalize();
    /// let normal_map_rgb = normal.remap(-1.0 .. 1.0, 0.0 .. 1.0);
    /// // or shorter
    /// let normal_map_rgb = normal.remap(-1..1, 0..1);
    /// // identical to
    /// let normal_map_rgb = normal * 0.5 + 0.5;
    /// ```
    #[track_caller]
    pub fn remap(self, from: impl VecRange<T, L>, to: impl VecRange<T, L>) -> Self
    where
        T: ScalarTypeFp,
    {
        let [a, b] = [from.get_bounds(), to.get_bounds()].map(VecBounds::by_len::<T, L>);

        // diff_a = a1 - a0;
        // diff_b = b1 - b0;
        // (self - a0) * (diff_b / diff_a) + b0

        Context::try_with(call_info!(), |ctx| {
            use VecBoundsByLen as ByLen;

            // try generate minimal code
            if let (ByLen::X1([(a0, _), (a1, _)]), ByLen::X1([(b0, _), (b1, _)])) = (a, b) {
                let arr = [a0.as_any(), a1.as_any(), b0.as_any(), b1.as_any()];
                let arr = {
                    let nodes = ctx.pool(); // minimal scope
                    arr.map(|any| any.try_eval_floating_point(&nodes))
                };
                if let [Some(a0), Some(a1), Some(b0), Some(b1)] = arr {
                    return self.remap_literals([a0, a1], [b0, b1], T::SCALAR_TYPE);
                }
            }

            // fallback to non optimal codegen

            // let diff_a = a1 - a0;
            // let diff_b = b1 - b0;
            // let fac = diff_b / diff_a;
            // return self * fac - (a0 * fac) + b0

            let (a0, b0, fac) = match (a, b) {
                (ByLen::X1([(a0, _), (a1, _)]), ByLen::X1([(b0, _), (b1, _)])) => {
                    let diff_a = a1 - a0;
                    let diff_b = b1 - b0;
                    let fac = diff_b / diff_a;
                    return self * fac - (a0 * fac).splat() + b0.splat();
                }
                (ByLen::L([(a0, _), (a1, _)]), ByLen::L([(b0, _), (b1, _)])) => {
                    let diff_a = a1 - a0;
                    let diff_b = b1 - b0;
                    let fac = diff_b / diff_a;
                    (a0, b0, fac)
                }
                (ByLen::L([(a0, _), (a1, _)]), ByLen::X1([(b0, _), (b1, _)])) => {
                    let diff_a = a1 - a0;
                    let diff_b = b1 - b0;
                    let fac = diff_b.splat() / diff_a;
                    (a0, b0.splat(), fac)
                }
                (ByLen::X1([(a0, _), (a1, _)]), ByLen::L([(b0, _), (b1, _)])) => {
                    let diff_a = a1 - a0;
                    let diff_b = b1 - b0;
                    let fac = diff_b / diff_a.splat();
                    (a0.splat(), b0, fac)
                }
            };

            self.mul_each(fac) - (a0.mul_each(fac)) + b0
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding).into())
    }

    /// restrict `self` to the `range` (half-open ranges are allowed too).
    ///
    /// If `self` lies outside the range `range`,
    /// returns the value closest to self within `range`.
    /// Otherwise returns `self`
    ///
    /// ## usage
    /// - `x.clamp(l..=r)`: returns `r` if `x > r`, `l` if `x < l`, otherwise `x`
    /// - `x.clamp(l..)`: `x.max(l)`
    /// - `x.clamp(..=r)`: `x.min(r)`
    ///
    /// see https://www.w3.org/TR/WGSL/#clamp
    #[track_caller]
    pub fn clamp(self, range: impl VecRangeBoundsInclusive<T, L>) -> Self
    where
        T: ScalarTypeNumber,
    {
        let self_any = self.as_any();
        match range.get_opt_bounds_inclusive().into_full_len_vecs::<T, L>() {
            [Some(s), Some(e)] => self_any.clamp(s.as_any(), e.as_any()),
            [None, Some(e)] => self_any.min(e.as_any()),
            [Some(s), None] => self_any.max(s.as_any()),
            [None, None] => self_any,
        }
        .into()
    }


    /// component wise max
    ///
    /// same as `self.clamp(other..)`
    ///
    /// compares the components in `self` and `other` pair wise and
    /// returns the larger of the two.
    ///
    /// see <https://www.w3.org/TR/WGSL/#max-float-builtin>
    #[track_caller]
    pub fn max(self, other: impl To<vec<T, L>>) -> Self
    where
        T: ScalarTypeNumber,
    {
        self.clamp(other..)
    }

    /// component wise min
    ///
    /// same as `self.clamp(..=other)`
    ///
    /// compares the components in `self` and `other` pair wise and
    /// returns the smaller of the two.
    ///
    /// see <https://www.w3.org/TR/WGSL/#min-float-builtin>
    #[track_caller]
    pub fn min(self, other: impl To<vec<T, L>>) -> vec<T, L>
    where
        T: ScalarTypeNumber,
    {
        self.clamp(..=other)
    }

    /// component wise floor
    ///
    /// rounds every component down to the closest integer that is <= itself
    ///
    /// more precisely:
    /// see <https://www.w3.org/TR/WGSL/#floor-expression>
    #[track_caller]
    pub fn floor(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().floor().into()
    }

    /// component wise ceiling
    ///
    /// rounds every component up to the closest integer that is >= itself
    ///
    /// more precisely:
    /// see <https://www.w3.org/TR/WGSL/#ceil-expression>
    #[track_caller]
    pub fn ceil(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().ceil().into()
    }

    /// difference to floor, also known as `fract` in shader languages.
    ///
    /// computes the same value as `self - self.floor()`
    ///
    /// > note: the name `fract` was changed because rusts `std::f32::fract` is
    /// > defined differently from the `fract` function in shaders.
    /// > To avoid headache when porting `glsl`/`wgsl` code to `shame` or
    /// > `shame` code to metaprogramming rust code, `shame` offers both variants
    /// > under different names:
    /// > * [`vec::dfloor`] "delta floor" (the common shader function)
    /// > * [`vec::sfract`] "signed fract" (the rust definition of fract)
    ///
    /// see <https://www.w3.org/TR/WGSL/#fract-builtin>
    #[track_caller]
    pub fn dfloor(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().fract().into()
    }

    /// signed fractional part
    ///
    /// this function corresponds to rusts [`std::f32::fract`] function
    /// which is defined as `self - self.trunc()`
    ///
    /// > note: the name `fract` was changed because rusts `std::f32::fract` is
    /// > defined differently from the `fract` function in shaders.
    /// > To avoid headache when porting `glsl`/`wgsl` code to `shame` or
    /// > `shame` code to metaprogramming rust code, `shame` offers both variants
    /// > under different names:
    /// > * [`vec::dfloor`] "delta floor" (the common shader function)
    /// > * [`vec::sfract`] "signed fract" (the rust definition of fract)
    ///
    /// see <https://www.w3.org/TR/WGSL/#fract-builtin>
    #[track_caller]
    pub fn sfract(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self - self.trunc()
    }

    /// ## use `dfloor` instead
    ///
    /// The name `fract` was changed because rusts `std::f32::fract` is
    /// defined differently from the `fract` function in shaders.
    /// To avoid headache when porting `glsl`/`wgsl` code to `shame` or
    /// `shame` code to metaprogramming rust code, `shame` offers both variants
    /// under different names:
    /// * [`vec::dfloor`] "delta floor" (the common shader function)
    /// * [`vec::sfract`] "signed fract" (the rust definition of fract)
    ///
    #[deprecated(note = "use `dfloor` instead")]
    pub fn fract_(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.dfloor()
    }

    /// component wise absolute value
    ///
    /// if a component in `self` is negative, returns a version of `self` where that component is positive instead
    ///
    /// For every component e in `self`, if e is a floating-point type,
    /// then the result is e with a positive sign bit.
    /// If e is an unsigned integer scalar type, then the result is e. If e is a
    /// signed integer scalar type and evaluates to the largest negative value,
    /// then the result is e.
    ///
    /// see <https://www.w3.org/TR/WGSL/#abs-float-builtin>
    #[track_caller]
    pub fn abs(self) -> vec<T, L>
    where
        T: ScalarTypeNumber,
    {
        self.as_any().abs().into()
    }

    /// component wise sign function
    ///
    /// for every component `c` in `self`, the corresponding component in the result vector is:
    /// * `1` when `c > 0`
    /// * `0` when `c == 0`
    /// * `-1` when `c < 0`
    ///
    /// see <https://www.w3.org/TR/WGSL/#sign-builtin>
    #[track_caller]
    pub fn sign(self) -> vec<T, L>
    where
        T: ScalarTypeSigned,
    {
        self.as_any().sign().into()
    }

    /// component wise rounding
    ///
    /// for every component e in `self`,
    /// returns the integer k nearest to e, as a floating point value.
    /// When a component e lies halfway between integers k and k + 1,
    /// the result is k when k is even, and k + 1 when k is odd.
    ///
    /// see <https://www.w3.org/TR/WGSL/#round-builtin>
    #[track_caller]
    pub fn round_ties_even(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().round_ties_even().into()
    }

    /// shorthand for `self.clamp(0..=1)` also known as `saturate`
    ///
    /// see <https://www.w3.org/TR/WGSL/#saturate-float-builtin>
    #[track_caller]
    pub fn clamp01(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().saturate().into()
    }

    /// component wise step function
    ///
    /// (an alternative [`vec::step_each`] function exists for vector `edge` values that are applied to `self` component wise)
    ///
    /// for every component `e` in `edge`:
    /// Returns 0.0 if `self` <= `e`, otherwise `1.0`
    ///
    /// see <https://www.w3.org/TR/WGSL/#step-builtin>
    #[track_caller]
    pub fn step(self, edge: impl To<vec<T, x1>>) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.step_each(edge.splat())
    }

    /// component wise step function
    ///
    /// (an alternative [`vec::step`] function exists for scalar `edge` values that are applied to all components of `self`)
    ///
    /// for every component `s` in `self` and the corresponding component `e` in `edge`:
    /// Returns 0.0 if `s` <= `e`, otherwise `1.0`
    ///
    /// see <https://www.w3.org/TR/WGSL/#step-builtin>
    #[track_caller]
    pub fn step_each(self, edge: impl To<vec<T, L>>) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().step(edge.to_any()).into()
    }

    /// component wise truncate
    ///
    /// for every component, returns the nearest whole number whose
    /// absolute value is less than or equal to the absolute value of that component.
    #[track_caller]
    pub fn trunc(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().trunc().into()
    }

    /// Returns the component-wise smooth Hermite interpolation between 0 and 1.
    ///
    /// ## Example
    /// ```
    /// let color = shame::vec!(1.0, 1.0, 1.0, 0.5);
    /// let color = color.smoothstep_each(0.5..0.6);
    /// ```
    ///
    /// an alternative [`vec::smoothstep`] function exists for situations where
    /// `self` is scalar and `from`, `to` are either scalar or vector.
    ///
    /// for a given `range` of `from..to` the result is
    /// ```
    /// let t = ((self - from) / (to - from)).clamp(0.0 ..= 1.0);
    /// t * t * (3.0 - 2.0 * t)
    /// ```
    ///
    /// Qualitatively:
    ///
    /// for a given `range` of `from..to`:
    /// * When `from` < `to`,
    ///   the function is 0 for x below `from`, then smoothly rises until x reaches `to`, and remains at 1 afterward.
    /// * When `from` > `to`,
    ///   the function is 1 for x below `to`, then smoothly descends until x reaches `from`, and remains at 0 afterward.
    ///
    /// see <https://www.w3.org/TR/WGSL/#smoothstep-builtin>
    #[track_caller]
    pub fn smoothstep_each(self, range: impl VecRange<T, L>) -> Self
    where
        T: ScalarTypeFp,
    {
        match range.get_bounds().by_len::<T, L>() {
            VecBoundsByLen::X1([(from, _), (to, _)]) => {
                match L::LEN {
                    ir::Len::X1 => {
                        // self, from and to are scalar
                        let self_ = vec::<T, x1>::from(self.as_any());
                        let scalar_result = self_.smoothstep_each_impl(from, to);
                        scalar_result.as_any().into()
                    }
                    _ => {
                        // self is vector, from and to are scalar
                        self.smoothstep_each_impl(from.splat(), to.splat())
                    }
                }
            }
            VecBoundsByLen::L([(from, _), (to, _)]) => self.smoothstep_each_impl(from, to),
        }
    }

    #[track_caller]
    fn smoothstep_each_impl(&self, from: Self, to: Self) -> Self
    where
        T: ScalarTypeFp,
    {
        self.as_any().smoothstep(from.to_any(), to.to_any()).into()
    }

    /// Returns the component wise linear blend of `from` and `to`
    /// where `self` is the vector of blend factors
    /// ```
    /// from * (vec::one() - self) + to * self
    /// ```
    /// this function is also known as `mix` (see <https://www.w3.org/TR/WGSL/#mix-builtin>).
    ///
    /// an alternative [`vec::lerp`] function exists for interpolating vectors with scalar blend factors
    ///
    #[track_caller]
    pub fn lerp_each(self, from: impl To<Self>, to: impl To<Self>) -> Self
    where
        T: ScalarTypeFp,
    {
        self.as_any().mix(from.to_any(), to.to_any()).into()
    }

    /// ## reverse subtraction
    ///
    /// returns `minuend - self`
    ///
    /// mostly useful for rewriting things like
    /// ```
    /// (1.0 - (1.0 - expression).func())
    /// ```
    /// as
    /// ```
    /// expression.rsub(1.0).func().rsub(1.0)
    /// ```
    /// -----
    /// ## rsub1()
    /// > note: theres also [`vec::rsub1`] for an even shorter shorthand in the 1.0 case:
    /// > ```
    /// > expression.rsub1().func().rsub1()
    /// > ```
    #[track_caller]
    pub fn rsub(self, minuend: impl To<vec<T, L>>) -> Self
    where
        T: ScalarTypeNumber,
    {
        minuend.to_gpu() - self
    }

    /// ## reverse subtraction of 1
    ///
    /// returns `1 - self`
    ///
    /// mostly useful for rewriting things like
    /// ```
    /// (1.0 - (1.0 - expression).func())
    /// ```
    /// as
    /// ```
    /// expresion.rsub1().func().rsub1()
    /// ```
    /// there is also the more general [`vec::rsub`] for values other than `1`
    #[track_caller]
    pub fn rsub1(self) -> Self
    where
        T: ScalarTypeNumber,
    {
        self.rsub(vec::one())
    }

    /// the euclidean distance between `self` and `other`
    ///
    /// returns the value `(self - other).length()`
    #[track_caller]
    pub fn distance(self, other: Self) -> vec<T, x1>
    where
        T: ScalarTypeFp,
    {
        self.as_any().distance(other.as_any()).into()
    }

    /// the dot product of `self` and `other`
    ///
    /// a · b
    /// is defined as
    /// (a.x * b.x) + (a.y * b.y) + ...
    ///
    /// the sum of the component wise products
    ///
    pub fn dot(self, other: Self) -> vec<T, x1>
    where
        T: ScalarTypeNumber,
    {
        match L::LEN {
            ir::Len::X1 => self.as_any() * other.as_any(),
            _ => self.as_any().dot(other.as_any()),
        }
        .into()
    }

    /// returns `self` if `a.dot(b)` is negative, and `-self` otherwise.
    #[track_caller]
    pub fn face_forward(self, a: impl To<vec<T, L>>, b: impl To<vec<T, L>>) -> vec<T, L>
    where
        T: ScalarTypeFp,
        L: AtLeastLen<x2>,
    {
        self.as_any().face_forward(a.to_any(), b.to_any()).into()
    }

    /// component wise inverse square root
    ///
    /// for every component `c` in `self`, computes `1 / √c`
    ///
    /// see <https://www.w3.org/TR/WGSL/#inverseSqrt-builtin>
    #[track_caller]
    pub fn inverse_sqrt(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().inverse_sqrt().into()
    }

    /// vector normalization
    ///
    /// returns a vector with length 1 and the same direction as `self`.
    ///
    /// If the length of `self` is zero (i.e. it has no direction)
    /// an indeterminate value is returned.
    ///
    /// see <https://www.w3.org/TR/WGSL/#normalize-builtin>
    #[track_caller]
    pub fn normalize(self) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        match L::LEN {
            ir::Len::X1 => self.sign(),
            _ => self.as_any().normalize().into(),
        }
    }


    /// reflected vector given the incident vector `self` and surface
    /// orientation `on_surface_with_normal`
    ///
    /// `v.reflect(nor)` returns the reflection direction `v - 2 * dot(nor, v) * nor`.
    ///
    /// see <https://www.w3.org/TR/WGSL/#reflect-builtin>
    #[track_caller]
    pub fn reflect(self, on_surface_with_normal: impl To<vec<T, L>>) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any().reflect(on_surface_with_normal.to_any()).into()
    }

    /// The function `inc.refract(nor, rior)` is defined as follows:
    ///
    /// for the incident vector `inc` and surface normal `nor`, and the ratio of
    /// indices of refraction `rior`,
    /// ```
    /// let k = 1.0 - rior * rior * (1.0 - dot(nor, inc) * dot(nor, inc)).
    /// ```
    /// If `k < 0.0`, returns the refraction zero vector,
    /// otherwise return the refraction vector
    /// ```
    /// rior * inc - (rior * dot(nor, inc) + sqrt(k)) * nor.
    /// ```
    /// The incident vector `inc` and the normal `nor` should be normalized for
    /// desired results according to Snell’s Law; otherwise, the results may
    /// not conform to expected physical behavior.
    ///
    /// see <https://www.w3.org/TR/WGSL/#refract-builtin>
    #[track_caller]
    pub fn refract(self, surface_normal: impl To<vec<T, L>>, ior_ratio: impl To<vec<T, x1>>) -> vec<T, L>
    where
        T: ScalarTypeFp,
    {
        self.as_any()
            .refract(surface_normal.to_any(), ior_ratio.to_any())
            .into()
    }

    /// fused multiply add
    ///
    /// `a.fma(b, c)` returns component wise `(a * b) + c`
    ///
    /// see <https://www.w3.org/TR/WGSL/#fma-builtin>
    #[track_caller]
    pub fn fma(self, multiply_by: impl To<Self>, then_add: impl To<Self>) -> Self
    where
        T: ScalarTypeFp,
    {
        self.as_any().fma(multiply_by.to_any(), then_add.to_any()).into()
    }

    /// the euclidean length of `self`
    ///
    /// Evaluates to `√(e.x² + e.y² + ...)` if `self` has multiple components
    ///
    /// https://www.w3.org/TR/WGSL/#length-builtin
    pub fn length(self) -> scalar<T>
    where
        T: ScalarTypeFp,
    {
        match L::LEN {
            ir::Len::X1 => self.as_any().abs(),
            _ => self.as_any().length(),
        }
        .into()
    }

    /// the squared length of `self`
    ///
    /// the value `self.length() * self.length()` is computed via `self.dot(self)`
    pub fn square_length(self) -> scalar<T>
    where
        T: ScalarTypeFp,
    {
        match L::LEN {
            ir::Len::X1 => self.as_any().abs().into(),
            _ => self.dot(self),
        }
    }

    /// returns the largest component of `self`
    pub fn max_comp(self) -> vec<T, x1>
    where
        T: ScalarTypeNumber,
    {
        self.reduce(vec::max)
    }

    /// returns the smallest component of `self`
    pub fn min_comp(self) -> vec<T, x1>
    where
        T: ScalarTypeNumber,
    {
        self.reduce(vec::min)
    }

    /// (no documentation yet)
    pub fn reduce(self, f: impl FnMut(vec<T, x1>, vec<T, x1>) -> vec<T, x1>) -> vec<T, x1> {
        self.into_iter().reduce(f).expect("no `x0: Len` exists")
    }

    /// returns the product of all components of `self`
    ///
    /// e.g. for 3 component vectors:
    ///
    /// `(self.x * self.y * self.z)`
    pub fn comp_product(self) -> vec<T, x1>
    where
        T: ScalarTypeNumber,
    {
        self.reduce(std::ops::Mul::mul)
    }

    /// returns the sum of all components of `self`
    ///
    /// e.g. for 3 component vectors:
    ///
    /// `(self.x + self.y + self.z)`
    #[track_caller]
    pub fn comp_sum(self) -> vec<T, x1>
    where
        T: ScalarTypeNumber,
    {
        self.reduce(std::ops::Add::add)
    }
}

impl<T: ScalarTypeFp> vec<T, x3> {
    /// cross product
    ///
    /// `a.cross(b)` returns a vector that is perpendicular to both `a` and `b`.
    /// The length of the returned vector equals the area of the parallelogram
    /// with `a` and `b` vectors as its sides.
    ///
    /// The result is equivalent to
    /// ```
    /// vec!(
    ///    a.y * b.z - a.z * b.y,
    ///    a.z * b.x - a.x * b.z,
    ///    a.x * b.y - a.y * b.x,
    /// )
    /// ```
    ///
    #[track_caller]
    pub fn cross(self, rhs: Self) -> Self { self.as_any().cross(rhs.as_any()).into() }
}

impl<T: ScalarTypeFp> vec<T, x2> {
    /// `atan2` of the vector components `y`, `x`.
    ///
    /// Returns an angle, in radians, in the interval [-π, π] whose tangent is y÷x.
    /// The quadrant selected by the result depends on the signs of y and x.
    ///
    /// see https://www.w3.org/TR/WGSL/#atan2-builtin
    #[track_caller]
    pub fn atan2(self) -> vec<T, x1> { vec::atan2_each(self.y, self.x) }
}

impl<T: ScalarTypeFp, L: Len> vec<T, L> {
    /// split the floating point components into fraction and exponent.
    /// (also known as `frexp`)
    ///
    /// returns `(fraction, exponent)` pair
    ///
    /// see https://www.w3.org/TR/WGSL/#frexp-builtin
    #[track_caller]
    pub fn fract_exp(self) -> (vec<T, L>, vec<i32, L>) {
        let fx = self.as_any().frexp(FrexpGenerics(T::SCALAR_TYPE_FP, L::LEN));
        (fx.fract.into(), fx.exp.into())
    }

    /// split the floating point components into fraction and whole part.
    /// (also known as `modf`)
    ///
    /// returns `(fraction, whole)` pair
    ///
    /// see https://www.w3.org/TR/WGSL/#modf-builtin
    #[track_caller]
    pub fn fract_whole(self) -> (vec<T, L>, vec<T, L>) {
        let fw = self.as_any().modf(ModfGenerics(T::SCALAR_TYPE_FP, L::LEN));
        (fw.fract.into(), fw.whole.into())
    }
}

impl<T: ScalarTypeNumber> vec<T, x3> {
    /// returns the median of the vector components `self.x`, `self.y`, `self.z`
    pub fn comp_median(self) -> vec<T, x1> {
        let [x, y, z] = [self.x, self.y, self.z];
        x.min(y).max(x.max(y).min(z))
    }
}

impl<L: Len> vec<bool, L> {
    /// whether all vector components are equal to `true`
    pub fn all(self) -> vec<bool, x1> { self.as_any().all().into() }

    /// whether at least one vector component is equal to `true`
    pub fn any(self) -> vec<bool, x1> { self.as_any().any().into() }

    /// component wise selection
    ///
    /// for every component `c` the ouptut vector's component `output.c` is computed as:
    /// ```
    /// output.c = if self.c { if_true.c } else { if_false.c }
    /// ```
    pub fn select_each<T: ScalarType>(self, if_true: vec<T, L>, if_false: vec<T, L>) -> vec<T, L> {
        self.as_any().select(if_true.as_any(), if_false.as_any()).into()
    }
}

impl vec<bool, x1> {
    /// component wise selection
    ///
    /// returns `if self { if_true } else { if_false }`
    pub fn select<T: ScalarType, L: Len>(self, if_true: vec<T, L>, if_false: vec<T, L>) -> vec<T, L> {
        self.as_any().select(if_true.as_any(), if_false.as_any()).into()
    }

    /// record an if statement, which gets turned into
    /// an `if (cond) { body }` statement within the generated shader.
    /// This is equivalent to [`shame::if_`]
    ///
    /// note: in the code generation step, the `then` function gets executed exactly once to
    /// record everything inside it.
    ///
    /// ## example
    /// ```
    /// use shame as sm;
    ///
    /// let i = sm::Cell::new(1u32);
    ///
    /// let condition = vertex.index.to_u32().gt(16u32);
    /// condition.then(move || {
    ///     // do something
    ///     i.set_add(4u32);
    /// });
    /// ```
    ///
    /// [`shame::if_`]: crate::if_
    #[track_caller]
    pub fn then(self, then: impl FnOnce() + FlowFn) {
        let r = IfRecorder::new().next(self.as_any());
        then();
        r.finish()
    }

    /// record a `if (cond) { discard; }` statement
    ///
    /// turns all currently active fragment threads that satisfy
    /// the `self` condition permanently inactive
    /// (that is, until the pipeline finishes executing).
    ///
    /// see [`discard_if`] for more info
    #[track_caller]
    pub fn then_discard(self) { discard_if(self); }
}

// overloads of BitCast, taken from https://www.w3.org/TR/WGSL/#bitcast-builtin
//
// for all these, imagine a where clause:
//
// (the scalar type `T` is provided in `.0`)
// where
//   S: ScalarTypeNumber + Is32Bit,
//   T: ScalarTypeNumber + Is32Bit, // .0
//   N: Len,
//
//
// (#0) fn bitcast<  TxN>(self:   SxN) ->   TxN
// (#1) fn bitcast<  Tx1>(self: f16x2) ->   Tx1
// (#2) fn bitcast<  Tx2>(self: f16x4) ->   Tx2
// (#3) fn bitcast<f16x2>(self:   Tx1) -> f16x2
// (#4) fn bitcast<f16x4>(self:   Tx2) -> f16x4

impl<T: ScalarType32Bit, L: Len> vec<T, L> {
    /// (no documentation yet)
    pub fn into_bits<R: ScalarType32Bit>(self) -> vec<R, L> {
        // bitcast overload #0, as counted in the large comment above
        self.as_any().bitcast(R::SCALAR_TYPE).into()
    }
}

impl vec<f16, x2> {
    /// (no documentation yet)
    pub fn into_bits<R: ScalarType32Bit>(self) -> vec<R, x1> {
        // bitcast overload #1, as counted in the large comment above
        self.as_any().bitcast(R::SCALAR_TYPE).into()
    }
}

impl vec<f16, x4> {
    /// (no documentation yet)
    pub fn into_bits_32<R: ScalarType32Bit>(self) -> vec<R, x2> {
        // bitcast overload #2, as counted in the large comment above
        self.as_any().bitcast(R::SCALAR_TYPE).into()
    }
}

impl<T: ScalarType32Bit> vec<T, x1> {
    /// (no documentation yet)
    pub fn into_bits_f16<R: ScalarType32Bit>(self) -> vec<f16, x2> {
        // bitcast overload #3, as counted in the large comment above
        self.as_any().bitcast(R::SCALAR_TYPE).into()
    }
}

impl<T: ScalarType32Bit> vec<T, x2> {
    /// (no documentation yet)
    pub fn into_bits_f16<R: ScalarType32Bit>(self) -> vec<f16, x4> {
        // bitcast overload #4, as counted in the large comment above
        self.as_any().bitcast(R::SCALAR_TYPE).into()
    }
}

impl<L: Len> vec<f32, L> {
    /// accepts vectors of any length and expands them to `x4`
    /// by adding default values for `x`(0.0), `y`(0.0), `z`(0.0) and `w`(1.0)
    ///
    /// vectors of different sizes get expanded as follows:
    /// - `f32x1` => `vec!(self.x, 0.0, 0.0, 1.0)`
    /// - `f32x2` => `vec!(self.xy(), 0.0, 1.0)`
    /// - `f32x3` => `vec!(self.xyz(), 1.0)`
    /// - `f32x4` => `self`
    ///
    #[track_caller]
    pub(crate) fn ext_homo(self) -> vec<f32, x4> {
        let self_ = self.as_any();
        let l4 = ir::Len::X4;
        let t = <Self as ToVec>::T::SCALAR_TYPE;
        match L::LEN {
            ir::Len::X1 => Any::new_vec(l4, t, &[self_, 0.0.to_any(), 0.0.to_any(), 1.0.to_any()]),
            ir::Len::X2 => Any::new_vec(l4, t, &[self_, 0.0.to_any(), 1.0.to_any()]),
            ir::Len::X3 => Any::new_vec(l4, t, &[self_, 1.0.to_any()]),
            ir::Len::X4 => self_,
        }
        .into()
    }

    /// component wise f16 quantization
    ///
    /// Quantizes 32-bit floating point values e as if e were converted to
    /// IEEE-754 binary16, and then converted back to IEEE-754 binary32.
    ///
    /// for more details,
    /// see <https://www.w3.org/TR/WGSL/#quantizeToF16-builtin>
    #[track_caller]
    pub fn quantize_to_f16(self) -> Self { self.as_any().quantize_to_f16().into() }
}
