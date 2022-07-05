//! swizzle functionality for tensors. `tensor.xyz()`, `tensor.xyz_mut()` etc.
use shame_graph::Any;
use super::*;

// swizzle impls
const X: u8 = 0;
const Y: u8 = 1;
const Z: u8 = 2;
const W: u8 = 3;

impl<S: IsVecOfAtLeast2, D: DType> Ten<S, D> {
    /// mutable lvalue referencing the vector's x component
    /// 
    /// `vector.x_mut().set(value)`
    pub fn x_mut(&mut self) -> &mut Ten<scal, D> {self.swizzle.reset_and_get_mut(self.any.x(), self.stage)}
    /// mutable lvalue referencing the vector's y component
    /// 
    /// `vector.y_mut().set(value)`
    pub fn y_mut(&mut self) -> &mut Ten<scal, D> {self.swizzle.reset_and_get_mut(self.any.y(), self.stage)}

    /// a vector's x component
    pub fn x(&self) -> Ten<scal, D> {Ten::from_downcast(self.any.x(), self.stage)}
    /// a vector's y component
    pub fn y(&self) -> Ten<scal, D> {Ten::from_downcast(self.any.y(), self.stage)}

    /// a 2 component vector made up of `self`'s x and y component
    pub fn xy(&self) -> Ten<vec2, D> {Ten::from_downcast(self.any.swizzle(&[X, Y]), self.stage)}

    /// a tuple made up of `self`'s x and y component
    pub fn x_y(&self) -> (Ten<scal, D>, Ten<scal, D>) {
        (self.x(), self.y())
    }

    /// a 3 component vector containing (x, y, 0)
    pub fn xy0(&self) -> Ten<vec3, D>  {
        (self.xy(), zero::<scal, D>()).vec()
    }

    /// a 4 component vector containing (x, y, 0, 0)
    pub fn xy00(&self) -> Ten<vec4, D> {
        (self.xy(), zero::<vec2, D>()).vec()
    }

    /// a 3 component vector containing (x, y, 1)
    pub fn xy1(&self) -> Ten<vec3, D> {
        (self.xy(), one::<scal, D>()).vec()
    }

    /// a 4 component vector containing (x, y, 0, 1)
    pub fn xy01(&self) -> Ten<vec4, D> {
        (self.xy(), zero::<scal, D>(), one::<scal, D>()).vec()    
    }
}

impl<S: IsVecOfAtLeast3, D: DType> Ten<S, D> {
    /// mutable lvalue referencing the vector's z component
    /// 
    /// `vector.z_mut().set(value)`
    pub fn z_mut(&mut self) -> &mut Ten<scal, D> {self.swizzle.reset_and_get_mut(self.any.z(), self.stage)}

    /// a vector's z component
    pub fn z(&self) -> Ten<scal, D> {Ten::from_downcast(self.any.z(), self.stage)}
    
    /// a 2 component vector made up of `self`'s y and z components
    pub fn yz(&self) -> Ten<vec2, D> {Ten::from_downcast(self.any.swizzle(&[Y, Z]), self.stage)}
    /// a 2 component vector made up of `self`'s x and z components
    pub fn xz(&self) -> Ten<vec2, D> {Ten::from_downcast(self.any.swizzle(&[X, Z]), self.stage)}

    /// mutable lvalue referencing the vector's x and y component as a 2 component vector
    /// 
    /// `vector.xy_mut().set(value)`
    pub fn xy_mut(&mut self) -> &mut Ten<vec2, D> {self.swizzle.reset_and_get_mut(self.any.swizzle(&[X, Y]), self.stage)}
    /// mutable lvalue referencing the vector's y and z component as a 2 component vector
    /// 
    /// `vector.yz_mut().set(value)`
    pub fn yz_mut(&mut self) -> &mut Ten<vec2, D> {self.swizzle.reset_and_get_mut(self.any.swizzle(&[Y, Z]), self.stage)}
    /// mutable lvalue referencing the vector's x and z component as a 2 component vector
    /// 
    /// `vector.xz_mut().set(value)`
    pub fn xz_mut(&mut self) -> &mut Ten<vec2, D> {self.swizzle.reset_and_get_mut(self.any.swizzle(&[X, Z]), self.stage)}

    /// a 3 component vector made up of `self`'s x, y, z components
    pub fn xyz(&self) -> Ten<vec3, D> {Ten::from_downcast(self.any.swizzle(&[X, Y, Z]), self.stage)}
    
    /// shorthand for `(self.x(), self.y(), self.z())`
    pub fn x_y_z(&self) -> (Ten<scal, D>, Ten<scal, D>, Ten<scal, D>) {
        (self.x(), self.y(), self.z())
    }

    /// shorthand for `(self.xy(), self.z())`
    pub fn xy_z(&self) -> (Ten<vec2, D>, Ten<scal, D>) {
        (self.xy(), self.z())
    }

    /// shorthand for `(self.x(), self.yz())`
    pub fn x_yz(&self) -> (Ten<scal, D>, Ten<vec2, D>) {
        (self.x(), self.yz())
    }

    /// a 4 component vector containing (x, y, z, 0)
    pub fn xyz0(&self) -> Ten<vec4, D> {
        (self.xyz(), zero()).vec()
    }

    /// a 4 component vector containing (x, y, z, 1)
    pub fn xyz1(&self) -> Ten<vec4, D> {
        (self.xyz(), one()).vec()
    }
}

impl<D: DType> Ten<vec4, D> {
    /// mutable lvalue referencing the vector's w component
    /// 
    /// `vector.w_mut().set(value)`
    pub fn w_mut(&mut self) -> &mut Ten<scal, D> {self.swizzle.reset_and_get_mut(self.any.w(), self.stage)}

    /// a vector's w component
    pub fn w(&self) -> Ten<scal, D> {Ten::from_downcast(self.any.w(), self.stage)}

    /// mutable lvalue referencing the vector's z and w component as a 2 component vector
    /// 
    /// `vector.zw_mut().set(value)`
    pub fn zw_mut(&mut self) -> &mut Ten<vec2, D> {self.swizzle.reset_and_get_mut(self.any.swizzle(&[Z, W]), self.stage)}
    
    /// a 2 component vector made up of `self`'s z and w components
    pub fn zw(&self) -> Ten<vec2, D> {Ten::from_downcast(self.any.swizzle(&[Z, W]), self.stage)}

    /// mutable lvalue referencing the vector's x, y, z components as a 3 component vector
    /// 
    /// `vector.xyz_mut().set(value)`
    pub fn xyz_mut(&mut self) -> &mut Ten<vec3, D> {self.swizzle.reset_and_get_mut(self.any.swizzle(&[X, Y, Z]), self.stage)}
    /// mutable lvalue referencing the vector's y, z, w components as a 3 component vector
    /// 
    /// `vector.yzw_mut().set(value)`
    pub fn yzw_mut(&mut self) -> &mut Ten<vec3, D> {self.swizzle.reset_and_get_mut(self.any.swizzle(&[Y, Z, W]), self.stage)}

    /// a 3 component vector made up of `self`'s y, z, w components
    pub fn yzw(&self) -> Ten<vec3, D> {Ten::from_downcast(self.any.swizzle(&[Y, Z, W]), self.stage)}

    /// shorthand for `(self.x(), self.y(), self.z(), self.w())`
    pub fn x_y_z_w(&self) -> (Ten<scal, D>, Ten<scal, D>, Ten<scal, D>, Ten<scal, D>) {
        (self.x(), self.y(), self.z(), self.w())
    }

    /// shorthand for `(self.xyz(), self.w())`
    pub fn xyz_w(&self) -> (Ten<vec3, D>, Ten<scal, D>) {
        (self.xyz(), self.w())
    }

    /// shorthand for `(self.x(), self.yzw())`
    pub fn x_yzw(&self) -> (Ten<scal, D>, Ten<vec3, D>) {
        (self.x(), self.yzw())
    }

    /// shorthand for `(self.xy(), self.zw())`
    pub fn xy_zw(&self) -> (Ten<vec2, D>, Ten<vec2, D>) {
        (self.xy(), self.zw())
    }
}

/// every shape - but mainly vecs (e.g. vec3) - define a corresponding 
/// `SwizzleMembers` enum type, which contains variants for every sub-shape 
/// smaller than that vec (e.g. vec2, scal). this SwizzleMembers type is then 
/// added as a Member to Ten<S, D>, and can therefore not contain Ten<S, D> itself 
/// (because that would make the type infinitely sized). Thats why SwizzleMember
/// types only contain sub-shapes smaller than its own. when handing out a 
/// `&mut Ten<SubShape, DType>`, the `SwizzleMembers` enum Variant of that 
/// SubShape and `DType` is filled with the corresponding [`Ten`].
macro_rules! define_shape_structs {
    ($($vecN: ident -> ($col: ty | $row: ty) $shape: expr, $num_comps: expr, $swizzle_vecN: ident($($sub_shapes: ident),*);)*) => {$(
        impl Shape for super::shape::$vecN {
            const SHAPE: shame_graph::Shape = $shape;
            type SwizzleMembers = $swizzle_vecN;

            type Col = $col;
            type Row = $row;
            const NUM_COMPONENTS: usize = $num_comps;
        }
        
        #[allow(non_camel_case_types, missing_docs)] 
        #[derive(Copy, Clone)]
        pub enum $swizzle_vecN {
            Empty,
            $(
                $sub_shapes(SwizzleTensForShape<$sub_shapes>),
            )*
        }

        impl IsSwizzleMembers for $swizzle_vecN {

            fn new_empty() -> Self {
                Self::Empty
            }

            // TODO: add an integrity check of the currently instanciated swizzle enum variant, 
            // this integrity check must be performed not only on mutable member acess such as 
            // `val.x_mut()` but also on `val.x()` since the user might have accidentially overwritten 
            // `val.x_mut() = 1.0.rec()` and then tries to read it via `val.x()`
            #[allow(unused)]
            fn reset_and_get_mut<SubS: Shape, D: DType>(&mut self, any: Any, stage: Stage) -> &mut Ten<SubS, D> {

                *self = match SubS::SHAPE {
                    $(
                        $sub_shapes::SHAPE => Self::$sub_shapes(reset_swizzle_member::<$sub_shapes, D>(any, stage)),
                    )*
                    _ => unreachable!("branch unreachable at reset swizzle member")
                };
                
                let rec: &mut Ten<SubS, D> = match self {
                    $(
                        Self::$sub_shapes(x) => {get_swizzle_member_mut(x)}
                    )*
                    _ => unreachable!("branch unreachable at get_mut member")
                };
            
                rec
           }
        }
    )*};
}

//see define_shape_structs macro_rules! definition for more info on whats going on here
define_shape_structs!{
  //          col   row  
    scal -> (scal | scal) shame_graph::Shape::Scalar, 1, swizzle_scal();
    vec2 -> (vec2 | scal) shame_graph::Shape::Vec(2), 2, swizzle_vec2(scal);
    vec3 -> (vec3 | scal) shame_graph::Shape::Vec(3), 3, swizzle_vec3(vec2, scal);
    vec4 -> (vec4 | scal) shame_graph::Shape::Vec(4), 4, swizzle_vec4(vec3, vec2, scal);
    mat2   -> (vec2 | vec2) shame_graph::Shape::Mat(2, 2), 2*2, swizzle_mat2();
    mat2x3 -> (vec2 | vec3) shame_graph::Shape::Mat(2, 3), 2*3, swizzle_mat2x3();
    mat3x2 -> (vec3 | vec2) shame_graph::Shape::Mat(3, 2), 3*2, swizzle_mat3x2();
    mat2x4 -> (vec2 | vec4) shame_graph::Shape::Mat(2, 4), 2*4, swizzle_mat2x4();
    mat4x2 -> (vec4 | vec2) shame_graph::Shape::Mat(4, 2), 4*2, swizzle_mat4x2();
    mat3   -> (vec3 | vec3) shame_graph::Shape::Mat(3, 3), 3*3, swizzle_mat3();
    mat3x4 -> (vec3 | vec4) shame_graph::Shape::Mat(3, 4), 3*4, swizzle_mat3x4();
    mat4x3 -> (vec4 | vec3) shame_graph::Shape::Mat(4, 3), 4*3, swizzle_mat4x3();
    mat4   -> (vec4 | vec4) shame_graph::Shape::Mat(4, 4), 4*4, swizzle_mat4();
}

/// swizzle helper types declared by define_shape_structs
pub trait IsSwizzleMembers: Copy {
    /// this will reinstantiate the swizzle Ten (in case the pervious &mut Ten that was given out was edited by the user) and returns a &mut Ten to it
    fn reset_and_get_mut<SubS: Shape, D: DType>(&mut self, any: Any, stage: Stage) -> &mut Ten<SubS, D>;
    /// constructs the initial empty state of `Self`
    fn new_empty() -> Self;
}

fn reset_swizzle_member<S: Shape, D: DType>(any: Any, stage: Stage) -> SwizzleTensForShape<S> {
    match D::DTYPE {
        shame_graph::DType::Bool => SwizzleTensForShape::<S>::Bool(Ten::from_downcast(any, stage)),
        shame_graph::DType::F32  => SwizzleTensForShape::<S>::F32 (Ten::from_downcast(any, stage)),
        shame_graph::DType::F64  => SwizzleTensForShape::<S>::F64 (Ten::from_downcast(any, stage)),
        shame_graph::DType::I32  => SwizzleTensForShape::<S>::I32 (Ten::from_downcast(any, stage)),
        shame_graph::DType::U32  => SwizzleTensForShape::<S>::U32 (Ten::from_downcast(any, stage)),
    }
}

fn get_swizzle_member_mut<S: Shape, S2: Shape, D: DType>(tens_for_shape: &mut SwizzleTensForShape<S2>) -> &mut Ten<S, D> {
    let err = "swizzle type-filter error";

    use SwizzleTensForShape as Tens;
    let swizzle = match tens_for_shape {
        Tens::F32 (ten) => try_same_mut(ten).expect(err),
        Tens::F64 (ten) => try_same_mut(ten).expect(err),
        Tens::U32 (ten) => try_same_mut(ten).expect(err),
        Tens::I32 (ten) => try_same_mut(ten).expect(err),
        Tens::Bool(ten) => try_same_mut(ten).expect(err),
    };
    swizzle
}

/// swizzle tensors for a given tensor shape `S`
#[derive(Clone, Copy)]
#[allow(missing_docs)]
pub enum SwizzleTensForShape<S: Shape> {
    F32 (Ten<S, f32 >),
    F64 (Ten<S, f64 >),
    U32 (Ten<S, u32 >),
    I32 (Ten<S, i32 >),
    Bool(Ten<S, bool>),
}

// returns `Some(a)` if `A` and `B` are the same type, `None` otherwise
fn try_same_mut<A: std::any::Any, B: 'static>(a: &mut A) -> Option<&mut B> {
    (a as &mut dyn std::any::Any).downcast_mut::<B>()
}