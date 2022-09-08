//! operators on recording types (mostly tensors), e.g. `+` `-` `/` `*` `+=`...
use shame_graph::Operator;

use super::*;
use std::ops::*;

macro_rules! impl_add_sub_div_operators {
    (
        $(
            ($($fXX: ty),*): ($Add: ident / $add: ident / $AddAssign: ident / $add_assign: ident) shape restriction: $shape_restrict: ident
        ;)*
    ) => {
        $(//ops unroll
            //impl for all Ten types and all Rhs possibilities
            impl<LhsS: $shape_restrict, Rhs: AsTen> $Add<Rhs> for Ten<LhsS, Rhs::D> 
            where (LhsS, Rhs::S): ScalarOrSame {
                type Output = Ten<<(LhsS, Rhs::S) as ScalarOrSame>::Widest, Rhs::D>;

                fn $add(self, rhs: Rhs) -> Self::Output {
                    self.as_any().$add(rhs.into_any()).downcast((self, rhs).narrow_or_push_error())
                }
            }

            //impl for all Rhs possibilities
            impl<S: $shape_restrict, T: AsTen> $AddAssign<T> for Ten<S, T::D> 
            where T::S: IsScalarOr<S> {
                fn $add_assign(&mut self, rhs: T) {
                    self.stage = (*self, rhs).narrow_or_push_error();
                    self.as_any().$add_assign(rhs.into_any())
                }
            }

            $(//dtypes unroll
                //impl for rust primitive types as Lhs
                impl<S: $shape_restrict> $Add<Ten<S, $fXX>> for $fXX { type Output = Ten<S, $fXX>; //TODO: this impl is using the wrong DType if the features for disable f64 etc are enabled.
                    fn $add(self, rhs: Ten<S, $fXX>) -> Self::Output {
                        self.into_any().$add(rhs.as_any()).downcast((self, rhs).narrow_or_push_error())
                    }
                }
            )* //dtypes unroll

        )* //ops unroll
    }
}

impl_add_sub_div_operators!{
    (f32, i32, bool): (Add/add/AddAssign/add_assign)       shape restriction: Shape;
    (f32, i32, bool): (Sub/sub/SubAssign/sub_assign)       shape restriction: Shape;
    (f32, i32, bool): (Div/div/DivAssign/div_assign)       shape restriction: Shape;
         (i32): (Rem/rem/RemAssign/rem_assign)             shape restriction: IsShapeScalarOrVec;
         (i32): (BitAnd/bitand/BitAndAssign/bitand_assign) shape restriction: Shape;
         (i32): (BitOr /bitor /BitOrAssign /bitor_assign ) shape restriction: Shape;
}

impl<S: Shape, D: DType> Neg for Ten<S, D> {
    type Output = Ten<S, D>;

    fn neg(self) -> Self::Output {
        self.into_any().neg().downcast(self.stage())
    }
}

impl<S: IsShapeScalarOrVec> Not for Ten<S, bool> {
    type Output = Ten<S, bool>;

    fn not(self) -> Self::Output {
        use shame_graph::Shape::*;
        match S::SHAPE {
            Scalar => self.into_any().not(),
            Vec(_) => self.into_any().not_each(),
            Mat(_, _) => 
            unreachable!("calling `not` operator on matrix Ten type"),
        }
        .downcast(self.stage())
    }
}

impl<LhsS: Shape, Rhs: AsTen> MulAssign<Rhs> for Ten<LhsS, Rhs::D> 
where (LhsS, Rhs::S): CanBeMultiplied {
    
    fn mul_assign(&mut self, rhs: Rhs) {
        self.stage = (*self, rhs).narrow_or_push_error();
        self.into_any().mul_assign(rhs.into_any());
    }
}

impl<LhsS: Shape, Rhs: AsTen> Mul<Rhs> for Ten<LhsS, Rhs::D> 
where (LhsS, Rhs::S): CanBeMultiplied {
    type Output = Ten<<(LhsS, Rhs::S) as CanBeMultiplied>::Output, Rhs::D>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        self.into_any().mul(rhs.into_any()).downcast((self, rhs).narrow_or_push_error())
    }
}

macro_rules! impl_lhs_rust_primitive_type_mul {
    ($(Ten<_, $dtype: ty> * $fXX: ident,)*) => {$(
        impl<RhsS: Shape> Mul<Ten<RhsS, $dtype>> for $fXX 
        where {
            type Output = Ten<RhsS, $dtype>;

            fn mul(self, rhs: Ten<RhsS, $dtype>) -> Self::Output {
                let stage = narrow_stages_or_push_error([Stage::Uniform, rhs.stage]);
                self.new_literal_any().mul(rhs.as_any()).downcast(stage)
            }
        }
    )*};
}
impl_lhs_rust_primitive_type_mul!(
    Ten<_, f32> * f32, 
    Ten<_, i32> * i32,
    Ten<_, u32> * u32,
);

impl<S: Shape, D: DType> Ten<S, D> {

    pub(crate) fn binary_assign_op(&mut self, val: impl AsTen<S=S, D=D>, op_assign: Operator) {
        self.stage = (*self, val).narrow_or_push_error(); // self gets narrowed stage assigned
        self.into_any().binary_assign_op(val.into_any(), op_assign);
    }

    /// records an assignment `=` operator in the shader. This is necessary
    /// because rust does not support overloading of the `=` operator itself. 
    pub fn set(&mut self, val: impl AsTen<S=S, D=D>) {
        self.binary_assign_op(val, Operator::Assign);
    }

    /// records an assignment `=` operator in the shader. This is necessary
    /// because rust does not support overloading of the `=` operator itself. 
    /// 
    /// this function does the same as `set`, i couldn't settle on a naming yet
    pub fn assign(&mut self, val: impl AsTen<S=S, D=D>) {
        self.set(val);
    }

    /// create a copy of `self` in the shader.
    /// This will most likely be realized in the shader by declaring a new
    /// variable and assigning the value to it, then returning that new variable
    /// 
    /// Not to be confused with the behavior of the `Clone` or `Copy` traits, 
    /// which would merely clone the recording reference and have no effect on 
    /// the resulting shader
    pub fn copy(&self) -> Self {
        self.into_any().copy().downcast(self.stage)
    }
}