//! functions that will show up in the shader when called on tensors and other
//! recording types
use crate::assert::assert_string;

use super::*;
use shame_graph::Any;

impl float2 {
    /// return the arc-tangent of `(self.y() / self.x())`
    pub fn atan2(&self) -> float {
        Any::atan2(self.into_any().y(), self.into_any().x()).downcast(self.stage())
    }
}

impl<S: Shape, D: DType> Ten<S, D> {

    fn with_constant_diagonal(val: D) -> Self {
        use shame_graph::Shape::*;
        let literal = val.new_literal().into_any();
        match S::SHAPE {
            Scalar => literal,
            Vec(_) | Mat(_, _) => Any::new_tensor(
                shame_graph::Tensor::new(S::SHAPE, D::DTYPE), 
                &[literal]
            )
        }
        .downcast(Stage::Uniform)
    }
    
    /// tensor filled with zeroes
    pub fn zero() -> Self {zero()}
    /// identity tensor wrt to `*` operator
    pub fn id()   -> Self {id()}
    /// tensor filled with ones
    pub fn one()  -> Self {one()}
}

impl<S: Shape, D: DType> Ten<S, D> where Self: std::ops::Mul<Output=Self> {
    /// raises `self` to the power of `n` by repeatedly multiplying it by itself.
    /// This actually records `n` multiplications in the shader
    pub fn pow_unrolled(self, n: u32) -> Self {
        self.with_any(|any| any.pow_unrolled(n))
    }
}

impl<S: IsShapeScalarOrVec, D: DType> Ten<S, D> {

    /// component wise application of the `sign` function, which returns
    /// - `-1` for negative numbers, 
    /// - `+1` for positive numbers,
    /// - `0` for `0`
    pub fn sign(&self) -> Self {
        use shame_graph::DType::*;
        match D::DTYPE {
            F32 | F64 | I32 => self.into_any().sign(),
            U32 => self.into_any().min(Any::uint(1)),
            Bool => self.into_any().copy(),
        }
        .downcast(self.stage)
    }

    /// the absolute value of `self`, applied component wise for vectors
    pub fn abs(&self) -> Self {
        use shame_graph::DType::*;
        match D::DTYPE {
            F32 | F64 | I32 => self.into_any().abs(),
            U32 | Bool => self.into_any().copy(),
        }
        .downcast(self.stage)
    } 
}

// genFType aka genType
impl<SelfS: IsShapeScalarOrVec> Ten<SelfS, f32> {

    /// raises `self` to the power of `exponent`. Applied component wise for 
    /// vectors
    /// 
    /// these signatures might be easier to read:
    /// ```ignore
    /// fn pow(&self: floatN, exponent: floatN) -> floatN
    /// fn pow(&self: floatN, exponent: float ) -> floatN
    /// ```
    pub fn pow<S: IsScalarOr<SelfS>>(&self, exponent: impl AsTen<S=S, D=f32>) -> Self {
        let any = exponent.into_any();

        let matching_exp = if S::SHAPE == SelfS::SHAPE {
            any
        } else if S::SHAPE == scal::SHAPE {
            //Any::pow requires a matching tensor
            Any::new_tensor(shame_graph::Tensor::new(SelfS::SHAPE, f32::DTYPE), &[any])
        } else {
            panic!("pow exponent shape is neither scalar nor {:?}", SelfS::SHAPE)
        };

        self.into_any().pow(matching_exp).downcast((*self, exponent).narrow_or_push_error())
    }

    /// applies the `eerp` function
    /// ```text
    ///     eerp(self, x, y) = x^(1-self)*y^(self) 
    /// ```
    /// component wise for vectors
    pub fn eerp<S: IsShapeScalarOrVec>(&self, x: impl AsTen<S=S, D=f32>, y: impl AsTen<S=S, D=f32>) 
    -> Ten<S, f32> 
    where SelfS: IsScalarOr<S> {
        let (x, y, t) = (x.as_ten(), y.as_ten(), *self);
        // eerp(t, x, y) = x * (y/x)^t
        let result = x.as_any() * (y/x).pow(t).as_any();
        result.downcast((x, y, t).narrow_or_push_error())
    }

    /// component wise `sin` function
    pub fn sin(&self) -> Self {
        self.into_any().sin().downcast(self.stage())
    }

    /// component wise `cos` function
    pub fn cos(&self) -> Self {
        self.into_any().cos().downcast(self.stage())
    }

    /// component wise `sin` function with its
    /// results remapped to the 0 to 1 range.
    pub fn sin01(&self) -> Self {
        self.sin().remap(-1.0..1.0, 0.0..1.0)
    }

    /// component wise `cos` function with its
    /// results remapped to the 0 to 1 range.
    pub fn cos01(&self) -> Self {
        self.sin().remap(-1.0..1.0, 0.0..1.0)
    }

    /// partial derivative of `self`'s `component`th component wrt neighboring
    /// fragments
    fn partial_derivative(&self, component: u8, precision: shame_graph::DerivativePrecision) -> Self {
        assert_string(matches!(self.stage, Stage::Fragment | Stage::Uniform), 
            "calling partial derivative function `dx`/`dy`/`dxy` on value that is neither uniform or fragment stage"
        );
        
        self.into_any().partial_derivative(component, precision).downcast(narrow_stages_or_push_error([self.stage, Stage::Fragment]))
    }

    /// partial derivative of `self` wrt to the neighboring fragment in 
    /// `x`-direction
    pub fn dx      (&self) -> Self {self.partial_derivative(0, shame_graph::DerivativePrecision::DontCare)}
    /// derivative of `self` wrt to the neighboring fragment in `y`-direction
    pub fn dy      (&self) -> Self {self.partial_derivative(1, shame_graph::DerivativePrecision::DontCare)}
    /// coarse-precision partial derivative of `self` wrt the neighboring fragment 
    /// in `x`-direction
    pub fn dx_coarse(&self) -> Self {self.partial_derivative(0, shame_graph::DerivativePrecision::Coarse)}
    /// coarse-precision partial derivative of `self` wrt the neighboring fragment 
    /// in `y`-direction
    pub fn dy_coarse(&self) -> Self {self.partial_derivative(1, shame_graph::DerivativePrecision::Coarse)}
    /// fine-precision partial derivative of `self` wrt the neighboring fragment 
    /// in `x`-direction
    pub fn dx_fine  (&self) -> Self {self.partial_derivative(0, shame_graph::DerivativePrecision::Fine)}
    /// fine-precision partial derivative of `self` wrt the neighboring fragment 
    /// in `y`-direction
    pub fn dy_fine  (&self) -> Self {self.partial_derivative(1, shame_graph::DerivativePrecision::Fine)}

    /// partial derivatives of `self` wrt to the neighboring fragments in 
    /// `x`-direction and `y`-direction
    pub fn dxy       (&self) -> (Self, Self) {(self.dx()       , self.dy())}
    /// coarse-precision partial derivatives of `self` wrt to the neighboring 
    /// fragments in `x`-direction and `y`-direction
    pub fn dxy_coarse(&self) -> (Self, Self) {(self.dx_coarse(), self.dy_coarse())}
    /// fine-precision partial derivatives of `self` wrt to the neighboring 
    /// fragments in `x`-direction and `y`-direction
    pub fn dxy_fine  (&self) -> (Self, Self) {(self.dx_fine()  , self.dy_fine())}
}

impl<SelfD: IsDTypeFloatingPoint> Ten<vec3, SelfD> {

    /// this signature might be easier to read:
    /// ```ignore
    /// fn cross(&self: floating3, val: floating3) -> floating
    /// ```
    /// the cross product of `self` and `val`
    pub fn cross(&self, val: impl AsTen<S=vec3, D=SelfD>) -> Ten<scal, SelfD> {
        self.into_any().cross(val.into_any()).downcast((*self, val).narrow_or_push_error())
    }
}

impl<SelfS: Shape, SelfD: IsDTypeNumber> Ten<SelfS, SelfD> {
    /// the sum of `self`'s components
    pub fn sum(&self) -> Ten<scal, SelfD> {
        self.components().sum()
    }
}

impl<SelfS: IsShapeScalarOrVec, SelfD: IsDTypeNumber> Ten<SelfS, SelfD> {

    /// these signatures might be easier to read:
    /// ```ignore
    /// fn clamp(&self: numberN, min: number , max: number ) -> numberN
    /// fn clamp(&self: numberN, min: numberN, max: numberN) -> numberN
    /// ```
    /// 
    /// returns 
    /// - `min` if `self < min`
    /// - `max` if `self > max`
    /// - `self` otherwise
    /// 
    /// applied component wise to vectors
    pub fn clamp<S: Shape>(&self, min: impl AsTen<S=S,D=SelfD>, max: impl AsTen<S=S,D=SelfD>) -> Self
    where S: IsScalarOr<SelfS> {
        self.into_any().clamp(min.into_any(), max.into_any()).downcast((*self, min, max).narrow_or_push_error())
    }

    /// these signatures might be easier to read:
    /// ```ignore
    /// fn min(&self: numberN, val: numberN) -> numberN
    /// fn min(&self: numberN, val: number ) -> numberN
    /// ```
    ///  returns 
    /// - `min` if `self < min`
    /// - `self` otherwise
    /// 
    /// applied component wise to vectors
    pub fn min<T: AsTen<D=SelfD>>(&self, other: T) -> Self
    where T::S: IsScalarOr<SelfS> {
        self.into_any().min(other.into_any()).downcast((*self, other).narrow_or_push_error())
    }

    /// performs `self.set(self.min(other))`
    pub fn min_assign<T: AsTen<D=SelfD>>(&mut self, other: T)
    where T::S: IsScalarOr<SelfS> {
        self.set(self.min(other))
    }

    /// these signatures might be easier to read:
    /// ```ignore
    /// fn max(&self: numberN, val: numberN) -> numberN
    /// fn max(&self: numberN, val: number ) -> numberN
    /// ```
    /// returns 
    /// - `max` if `self > max`
    /// - `self` otherwise
    /// 
    /// applied component wise to vectors
    pub fn max<T: AsTen<D=SelfD>>(&self, other: T) -> Self 
    where T::S: IsScalarOr<SelfS> {
        self.into_any().max(other.into_any()).downcast((*self, other).narrow_or_push_error())
    }

    /// performs `self.set(self.max(other))`
    pub fn max_assign<T: AsTen<D=SelfD>>(&mut self, other: T)
    where T::S: IsScalarOr<SelfS> {
        self.set(self.max(other))
    }

    fn compare(&self, kind: shame_graph::CompareKind, rhs: &impl AsTen<S=SelfS, D=SelfD>) -> Ten<SelfS, bool> {
        use shame_graph::{Shape::*, DType::*};
        if let (Vec(_), F64) = (SelfS::SHAPE, SelfD::DTYPE) {
            todo!("{:?} not yet implemented for {}<{}>", kind, SelfS::SHAPE, SelfD::DTYPE);
        }
        match SelfS::SHAPE {
            Scalar => self.into_any().scalar_comparison(kind, rhs.into_any()),
            Vec(_) => self.into_any().vector_comparison(kind, rhs.into_any()),
            Mat(_, _) => unreachable!("running `compare` with matrix arguments"),
        }.downcast((*self, *rhs).narrow_or_push_error())
    }

    /// component wise comparison of `self` with `rhs` using `==`
    pub fn eq(&self, rhs: &impl AsTen<S=SelfS, D=SelfD>) -> Ten<SelfS, bool> {
        self.compare(shame_graph::CompareKind::Equal, rhs)
    }
    
    /// component wise comparison of `self` with `rhs` using `!=`
    pub fn ne(&self, rhs: &impl AsTen<S=SelfS, D=SelfD>) -> Ten<SelfS, bool> {
        self.compare(shame_graph::CompareKind::NotEqual, rhs)
    }

    /// component wise comparison of `self` with `rhs` using `<`
    pub fn lt(&self, rhs: &impl AsTen<S=SelfS, D=SelfD>) -> Ten<SelfS, bool> {
        self.compare(shame_graph::CompareKind::Less, rhs)
    }

    /// component wise comparison of `self` with `rhs` using `<=`
    pub fn le(&self, rhs: &impl AsTen<S=SelfS, D=SelfD>) -> Ten<SelfS, bool> {
        self.compare(shame_graph::CompareKind::LessEqual, rhs)
    }

    /// component wise comparison of `self` with `rhs` using `>`
    pub fn gt(&self, rhs: &impl AsTen<S=SelfS, D=SelfD>) -> Ten<SelfS, bool> {
        self.compare(shame_graph::CompareKind::Greater, rhs)
    }

    /// component wise comparison of `self` with `rhs` using `>=`
    pub fn ge(&self, rhs: &impl AsTen<S=SelfS, D=SelfD>) -> Ten<SelfS, bool> {
        self.compare(shame_graph::CompareKind::GreaterEqual, rhs)
    }

} 

impl<SelfS: IsShapeScalarOrVec> Ten<SelfS, bool> {

    /// glsl equivalent: `mix` with `bool` alpha.
    /// 
    /// this signature might be easier to read:
    /// ```ignore
    /// fn select(&self: boolN, x: anyN, y: anyN) -> anyN
    /// ```
    pub fn select<D: DType>(&self, x: impl AsTen<S=SelfS, D=D>, y: impl AsTen<S=SelfS, D=D>) -> Ten<SelfS, D> {
        self.into_any().mix(x.into_any()..y.into_any()).downcast((*self, x, y).narrow_or_push_error())
    }
}

impl<SelfS: Shape, SelfD: IsDTypeFloatingPoint> Ten<SelfS, SelfD> {
    /// these signatures might be easier to read:
    /// ```ignore
    /// fn mix(&self: floatingN, x: floating , y: floating )
    /// fn mix(&self: floatingN, x: floatingN, y: floatingN)
    /// ```
    /// 
    /// linear interpolation or `lerp` from `x` to `y` with mix value `self`
    pub fn mix<S: IsShapeScalarOrVec>(&self, x: impl AsTen<S=S, D=SelfD>, y: impl AsTen<S=S, D=SelfD>) -> Ten<S, SelfD> 
    where SelfS: IsScalarOr<S> {
        self.into_any().mix(x.into_any()..y.into_any()).downcast((*self, x, y).narrow_or_push_error())
    }

    /// linear interpolation from `x` to `y` with mix value `self`. 
    /// Alternate function naming to `mix`, which is how `lerp` is called in glsl
    /// 
    /// these signatures might be easier to read:
    /// ```ignore
    /// fn mix(&self: floatingN, x: floating , y: floating )
    /// fn mix(&self: floatingN, x: floatingN, y: floatingN)
    /// ```
    pub fn lerp<S: IsShapeScalarOrVec>(&self, x: impl AsTen<S=S, D=SelfD>, y: impl AsTen<S=S, D=SelfD>) -> Ten<S, SelfD> 
    where SelfS: IsScalarOr<S> {
        self.mix(x, y)
    }

    /// component wise `floor` rounding of `self`
    pub fn floor(&self) -> Self {self.as_any().floor().downcast(self.stage)}
    /// component wise `ceil` rounding of `self`
    pub fn ceil (&self) -> Self {self.as_any().ceil ().downcast(self.stage)}
    /// component wise rounding of `self` to the closest integer.
    /// returns the same type as `self`
    /// 
    /// whether values like `0.5` round up or down is implementation defined
    pub fn round(&self) -> Self {self.as_any().round().downcast(self.stage)}

    /// arithmetic mean of all the components
    /// ```ignore
    /// // e.g. for v: float4
    /// (v.x + v.y + v.z) / 3.0
    /// ```
    pub fn avg(&self) -> Ten<scal, SelfD> {
        self.sum() / (self.components_len() as u32).cast::<SelfD>()
    }
}



impl<SelfS: IsShapeScalarOrVec, SelfD: IsDTypeFloatingPoint> Ten<SelfS, SelfD> {

    /// square root of self
    pub fn sqrt(&self) -> Self {
        self.into_any().sqrt().downcast(self.stage())
    }

    /// returns self - self.floor()
    pub fn fract(&self) -> Self {
        self.into_any().fract().downcast(self.stage())
    }

    /// euclidean length of the vector
    ///
    /// note: if you need the amount of components in vector instead, use 
    /// `ten.components_len()`
    pub fn length(&self) -> Ten<scal, SelfD> {
        self.into_any().length().downcast(self.stage())
    }

    /// returns the dot product of self with itself
    pub fn square_length(&self) -> Ten<scal, SelfD> {
        self.dot(*self)
    }

    /// returns a vector in the same direction as x but with a length of 1, 
    /// i.e. self / self.length()
    pub fn normalize(&self) -> Self {
        self.into_any().normalize().downcast(self.stage())
    }

    /// the dot product of `self` with `val`
    pub fn dot(&self, val: impl AsTen<S=SelfS, D=SelfD>) -> Ten<scal, SelfD> {
        self.into_any().dot(val.into_any()).downcast((*self, val).narrow_or_push_error())
    }

    /// these signatures might be easier to read:
    /// ```text
    /// fn remap(&self: floatingN, from: Range<floating >, to: Range<floating >) -> floatingN
    /// fn remap(&self: floatingN, from: Range<floatingN>, to: Range<floatingN>) -> floatingN
    /// ```
    /// ---
    /// `remap(from, to)` takes values that are assumed to be in range `from` 
    /// and scales/moves them to end up in the `to` range.
    /// This is useful for e.g. moving things from a `-1..1` range into `0..1` 
    /// and vice versa.
    /// ```text
    /// let normal = sampler.sample(uv).remap(0..1, -1..1);
    /// ```
    pub fn remap<S: IsScalarOr<SelfS>>(&self, 
            from: std::ops::Range<impl AsTen<S=S>>, 
              to: std::ops::Range<impl AsTen<S=S>>) -> Self {
        let a = from.into_recs();
        let b =   to.into_recs();

        let [a0, a1] = [a.0.cast::<SelfD>().as_any(), a.1.cast::<SelfD>().as_any()];
        let [b0, b1] = [b.0.cast::<SelfD>().as_any(), b.1.cast::<SelfD>().as_any()];

        let a_width = a1 - a0;
        let b_width = b1 - b0;

        let scale = b_width / a_width;

        let result = (self.into_any() + (-a0)) * scale + b0;
        result.downcast((*self, a.0, a.1, b.0, b.1).narrow_or_push_error())
    }

    /// these signatures might be easier to read:
    /// ```ignore
    /// fn limit(&self: floatingN, bounds: impl RangeBounds<floating >) -> floatingN
    /// fn limit(&self: floatingN, bounds: impl RangeBounds<floatingN>) -> floatingN
    /// ```
    /// ---
    /// limit is a generalization of `clamp, min, max` that supports 
    /// closed/half-open/open ranges.
    /// 
    /// note: `a..=b` and `a..b` are treated the same 
    /// (as start-inclusive, end-inclusive)
    /// ```text
    /// x.limit(btm..)    --> x.max(btm)
    /// x.limit(..top)    --> x.min(top)
    /// x.limit(btm..top) --> x.clamp(btm, top)
    /// ```
    pub fn limit<S: Shape, Limit: AsTen<S=S, D=SelfD>>(&self, bounds: impl std::ops::RangeBounds<Limit>) -> Self
    where S: IsScalarOr<SelfS> {
        use std::ops::Bound::*;
        match (bounds.start_bound(), bounds.end_bound()) {

            (Included(s)|Excluded(s), Included(e)|Excluded(e))  // s..e | s..=e
            => self.into_any().clamp(s.into_any(), e.into_any()).downcast((*self, *s, *e).narrow_or_push_error()),

            (Unbounded, Included(e)|Excluded(e)) // ..e | ..=e
            => self.into_any().min(e.into_any()).downcast((*self, *e).narrow_or_push_error()), 

            (Included(s)|Excluded(s), Unbounded) // e..
            => self.into_any().max(s.into_any()).downcast((*self, *s).narrow_or_push_error()),

            (Unbounded, Unbounded) //..
            => self.into_any().copy().downcast(self.stage())
        }
    }

    /// component wise application of the smoothstep function
    /// ```text
    /// let t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1);
    /// return t * t * (3 - 2 * t);
    /// ```
    pub fn smoothstep<EdgeS>(&self, edge0: impl AsTen<S=EdgeS, D=SelfD>, edge1: impl AsTen<S=EdgeS, D=SelfD>) -> Self 
    where EdgeS: IsScalarOr<SelfS> {
        self.into_any().smoothstep(edge0.into_any()..edge1.into_any()).downcast((*self, edge0, edge1).narrow_or_push_error())
    }

    /// same as smoothstep, but with a `std::ops::Range` argument which can be more convenient
    /// ```ignore
    /// val.smoothrange(0..1);
    /// val.smoothrange(0.5.plus_minus(0.1)); //same as 0.4..0.6
    /// ```
    pub fn smoothrange<S: IsShapeScalarOrVec>(&self, range: std::ops::Range<impl AsTen<S=S, D=SelfD>>) -> Self 
    where S: IsScalarOr<SelfS> {
        let (s, e) = (range.start.rec(), range.end.rec());
        self.into_any().smoothstep(s.as_any()..e.as_any()).downcast((*self, s, e).narrow_or_push_error())
    }

    /// generalized smoothstep function of order `order`.
    /// 
    /// `order = 0` records the builtin shader function `clamp` remapped to `0.0..1.0`.
    /// 
    /// `order = 1` records the builtin shader function `smoothstep`.
    /// 
    /// `order > 1` calculates the generalized smoothstep with arithmetic expressed in the shader
    pub fn smoothstep_n<S: IsShapeScalarOrVec>(&self, order: u32, range: std::ops::Range<impl AsTen<S=S, D=SelfD>>) -> Self
    where S: IsScalarOr<SelfS>,
    SelfS: IsScalarOr<SelfS> {
        //todo!("the remapping of this function is still wrong");
        let (s, e) = (range.start.as_ten(), range.end.as_ten());
        match order {
            0 => self.clamp(s, e).remap(range, zero_to_one::<S, SelfD>()),
            1 => self.smoothstep(s, e),
            _ => self.generalized_smoothstep(order).remap(range, zero_to_one::<S, SelfD>()),
        }
    }

    fn generalized_smoothstep(&self, order: u32) -> Self {
        // Returns binomial coefficient without explicit use of factorials,
        // which can't be used with negative integers
        let pascal_triangle = |a, b| (0..b).fold(1, |acc, i| acc * (a - i) / (i + 1));

        let n = order as i32;
        self.clamp(zero::<scal, SelfD>(), one()).with_any(|x| {
            (0..=n).fold(zero::<SelfS, SelfD>().into_any(), |acc, i| {
                let fac = pascal_triangle(-n - 1, i) * pascal_triangle(2 * n + 1, n - i);
                let fac = fac as f32;
                acc + fac.new_literal_any() * x.pow_unrolled((n + i + 1) as u32)
            })
        })
    }
}

impl<S: IsShapeVec> Ten<S, bool> {

    /// the glsl `bool all(bvec)` function, name was changed to prevent confusion
    pub fn fold_or(&self) -> Ten<scal, bool> {
        self.into_any().all().downcast(self.stage)
    }

    /// the glsl `bool any(bvec)` function, name was changed to prevent confusion
    pub fn fold_and(&self) -> Ten<scal, bool> {
        self.into_any().any_is().downcast(self.stage)
    }

    /// the glsl `bool all(bvec)` function
    /// fold_or/fold_and may offer a better naming scheme, use whatever you like better
    #[inline]
    pub fn all(&self) -> Ten<scal, bool> {
        self.fold_or()
    }

    /// the glsl `bool any(bvec)` function, name was changed to prevent confusion with the `Rec` function that returns the `Any` type.
    /// fold_or/fold_and may offer a better naming scheme, use whatever you like better
    #[inline]
    pub fn any_is(&self) -> Ten<scal, bool> {
        self.fold_and()
    }
    
}

impl float2 {

    /// rotate `self` around the origin by the angle `radians`
    pub fn rotate_2d(&self, radians: impl AsFloat) -> float2 {
        let ang = radians.as_ten();
        let (x, y) = self.x_y();
        (
            x * ang.cos() - y * ang.sin(),
            x * ang.sin() + y * ang.cos(),
        ).vec()
    }

    /// rotate `self` around `pivot` by the angle `radians`
    pub fn rotate_2d_around(&self, radians: impl AsFloat, pivot: impl AsFloat2) -> float2 {
        let pivot = pivot.rec();
        let rel = *self - pivot;
        let (x, y) = rel.x_y();
        
        let ang = radians.as_ten();
        let rotated = (
            x * ang.cos() - y * ang.sin(),
            x * ang.sin() + y * ang.cos(),
        ).vec();
        rotated + pivot
    }

}

/// returns a tensor with all components set to zero of any tensor type.
/// 
/// If you have trouble with type deduction you can use e.g. `float4::zero()`, `int3::zero()` etc.
pub fn zero<S: Shape, D: DType>() -> Ten<S, D> {
    Ten::with_constant_diagonal(D::from_f32(0.0))
}

/// multiplicative identity of `Ten<S, D>`
///
/// If you have trouble with type deduction you can use e.g. `float3x3::id()`, `int4::id()` etc.
/// 
/// returns 1 for scalars,
/// 
/// returns identity matrix with 1 at diagonal for matrix types (rest of components are set to zero)
pub fn id<S: Shape, D: DType>() -> Ten<S, D> {
    Ten::with_constant_diagonal(D::from_f32(1.0))
}

/// returns a tensor with all components set to one (even for matrices. use `id()` if you want only the diagonal of the matrix to be set)
/// 
/// If you have trouble with type deduction you can use e.g. `float4::one()`, `int3::one()` etc.
pub fn one<S: Shape, D: DType>() -> Ten<S, D> {
    use shame_graph::Shape::*;
    use shame_graph::Tensor;
    match S::SHAPE {
        Scalar | Vec(_) => id(),
        Mat(_, _) => {
            assert!(S::NUM_COMPONENTS <= 16);
            let ones = [(); 16].map(|_| D::from_f32(1.0).new_literal_any());
            Any::new_tensor(Tensor::new(S::SHAPE, D::DTYPE), &ones[..S::NUM_COMPONENTS])
            .downcast(Stage::Uniform)
        },
    }
}

/// generalized 0 to 1 range for any tensor [`DType`] and [`Shape`]
pub fn zero_to_one<S: Shape, D: DType>() -> std::ops::Range<Ten<S, D>> {
    zero::<S, D>()..one::<S, D>()
}

/// generalized -1 to +1 range for any tensor [`DType`] and [`Shape`]
pub fn minus_one_to_one<S: Shape, D: DType>() -> std::ops::Range<Ten<S, D>> {
    (-one::<S, D>())..one::<S, D>()
}

/// convenience trait for calculating ranges around a center
pub trait PlusMinus<Rhs=Self> {
    /// output range type. mostly `std::ops::Range<Self>`
    type Output;
    /// returns `(self - rhs)..(self + rhs)`
    fn plus_minus(&self, rhs: Rhs) -> Self::Output;
}

impl<Rhs: AsTen<D=f32>> PlusMinus<Rhs> for Ten<Rhs::S, f32> {
    type Output = std::ops::Range<Self>;
    fn plus_minus(&self, rhs: Rhs) -> Self::Output {
        (self.into_any() - rhs.into_any()).downcast((*self, rhs).narrow_or_push_error())
        ..
        (self.into_any() + rhs.into_any()).downcast((*self, rhs).narrow_or_push_error())
    }
}

impl<Rhs: AsFloat> PlusMinus<Rhs> for f32 {
    type Output = std::ops::Range<float>;
    fn plus_minus(&self, rhs: Rhs) -> Self::Output {
        (self.into_any() - rhs.into_any()).downcast((*self, rhs).narrow_or_push_error())
        ..
        (self.into_any() + rhs.into_any()).downcast((*self, rhs).narrow_or_push_error())
    }
}
