//! array recording types of different kinds (sized, runtime-sized, ...)
use std::{marker::PhantomData, rc::Rc};
use crate::{rec::{Rec, Stage, AsTen}, IsDTypeInteger};
use shame_graph::{Any, shorthands::new_array_enumerate, Access};

use super::{IntoRec, DType, scal, narrow_stages_or_push_error, fields::Fields};

/// types that describe array size information (or lack thereof) 
pub trait Sizing {
    /// contains the array length in case it is known at rust compile-time
    const CONST_LEN: Option<usize>;
}

/// an Array<T, Unsized> is commonly used for buffer bindings etc.
pub struct Unsized;
/// an Array<T, Size<N>> has known size at rust compile time and can 
/// conveniently be created from rust types such as `[T; N]` by calling `.rec()`
pub struct Size<const N: usize>;
/// an Array<T, RecordedSize> is not `shame::Rec`! It has a size that is 
/// known at shader compile time, but not at rust compile time.
/// Use this if the array size depends on some calculation you do while 
/// generating the shader. 
pub struct RecordedSize;

impl              Sizing for Unsized      {const CONST_LEN: Option<usize> = None;   }
impl<const N: usize> Sizing for Size<N>      {const CONST_LEN: Option<usize> = Some(N);}
impl              Sizing for RecordedSize {const CONST_LEN: Option<usize> = None;   }

/// recording type ([`Rec`]) that represents arrays in shaders. These arrays can
/// be indexed at shader runtime with [`Rec`] indices. 
/// 
/// `Len` can be either 
/// - [`Unsized`]: The length of the array is not known at shader runtime (this 
/// is commonly used for storage buffer bindings)
/// - [`Size<N>`]: The length of the array is known at rust compile time
/// - [`RecordedSize`]: The length of the array is decided when the shader is
/// recorded at rust-runtime, but it is known at shader-compile-time
#[derive(Clone, Copy)]
pub struct Array<T: Rec, Len: Sizing = Unsized> {
    any: Any,
    stage: Stage,
    last_mut_access: Option<T>, //this is for doing the same trick as with `Ten`'s mutable swizzle `.xy_mut()` etc.
    phantom: PhantomData<(T, Len)>,
}

impl<T: Rec, const N: usize> Array<T, Size<N>> {

    /// initialize a new sized [`Array`] array with `elements`
    /// 
    /// alternatively call `elements.rec()`
    pub fn new<X: IntoRec<Rec=T>>(elements: [X; N]) -> Self {
        let args: [T; N] = elements.map(|x|x.rec());
        let arg_anys: [Any; N] = new_array_enumerate(|i| args[i].as_any());

        let stage = narrow_stages_or_push_error(args.iter().map(|x|x.stage()));
        
        let array_ty = shame_graph::Array::new_sized(T::ty(), N);
        let any = Any::array_initializer(array_ty, &arg_anys);

        Array { 
            any,
            stage,
            last_mut_access: None,
            phantom: PhantomData,
        }
    }
    
    /// the amount of elements in the array
    pub fn len(&self) -> usize {
        N
    }

    /// indexing with a record-time constant, which means bounds checking is 
    /// possible, therefore this function is not unsafe.
    pub fn at_const(&self, i: u32) -> T {
        let n = N as u32;

        match (0..n).contains(&i) {
            true => unsafe {self.at(i)},
            false => panic!("shader recording: array index {i} out of bounds 0..{N}"),
        }
    }

    /// indexing with a record-time constant, which means bounds checking is 
    /// possible, therefore this function is not unsafe.
    pub fn at_const_mut(&mut self, i: u32) -> &mut T {
        let n = N as u32;

        match (0..n).contains(&i) {
            true => unsafe {self.at_mut(i)},
            false => panic!("shader recording: array index {i} out of bounds 0..{N}"),
        }
    }
}

impl<T: Rec, Len: Sizing> Array<T, Len> {

    fn check_last_mut_access_integrity(&self) {
        //TODO: implement
    }

    /// Mutable access to an array element at index `i`.
    /// The array access is not bounds checked, therefore this function is unsafe.
    pub unsafe fn at_mut<D: DType>(&mut self, i: impl AsTen<S=scal, D=D>) -> &mut T {
        self.check_last_mut_access_integrity();

        let (index_any, index_stage) = scalar_to_index(i.as_ten());
        //TODO: copy on write access doesn't really act like copy on write here, if an index is being manipulated afterwards, it is not written into a new variable. `i += 1` recorded on a i that is copy on write does not yield `int _ = i + 1`. Maybe CopyOnWrite needs to be renamed
        let index_any = index_any.copy().aka("_i"); //to prevent the user from changing the index after obtaining the &mut T from this function

        //TODO: can the problem from the previous todo maybe be resolved by introducing the following rule:
        // a subscript on a mutable array is itself a mutable operation.
        let any = self.any.subscript(index_any);
        debug_assert!(matches!(any.ty_via_thread_ctx().map(|x|x.access), None | Some(Access::LValue)));

        let t = T::from_downcast(any, narrow_stages_or_push_error([self.stage, index_stage]));
        self.last_mut_access.replace(t);
        match &mut self.last_mut_access {
            Some(t) => t,
            None => unreachable!(),
        }
    }

    /// *copies* the value of `array[i]` in the shader.
    /// The array access is not bounds checked, therefore this function is unsafe.
    /// 
    /// **note: this does *NOT* produce an lvalue, which means that**
    /// ```ignore
    /// array.at(i).set(value) //does NOT work as expected since glsl does not support lvalue references, and rust does not allow `=` operator overloading.
    /// array.at_mut(i).set(value) //produces the expected `array[i] = value` in the shader
    /// ```
    pub unsafe fn at<D: IsDTypeInteger>(&self, i: impl AsTen<S=scal, D=D>) -> T {
        let (any, stage) = scalar_to_index(i);
        T::from_downcast(self.any.subscript(any).copy().aka("_i"), narrow_stages_or_push_error([self.stage, stage]))
    }

    /// produces a subscript operation with assignment `array[i] = val` in the 
    /// shader
    /// The array access is not bounds checked, therefore this function is unsafe.
    pub unsafe fn set_at<D: IsDTypeInteger>(&mut self, i: impl AsTen<S=scal, D=D>, val: T) {
        let (any, stage) = scalar_to_index(i);
        narrow_stages_or_push_error([self.stage, stage, val.stage()]);
        self.any.subscript(any).assign(val.as_any())
    }

}

/// only `int` and `uint` can be used for indexing in shaders. this function 
/// casts other scalar types to `int` or `uint` and returns the type erased 
/// `Any` and `Stage` of it. When provided with a value that is already `int` 
/// or `uint`, no cast operation is recorded.
fn scalar_to_index<D: DType>(scalar: impl AsTen<S=scal, D=D>) -> (Any, Stage) {
    use shame_graph::DType::*;
    let ten = scalar.as_ten();
    let any = ten.as_any();
    let any = match D::DTYPE {
        Bool | F32 | F64 => Any::cast_uint(any),
        I32 | U32 => any,
    };
    (any, ten.stage())
}

impl<T: Rec, const N: usize> Rec for Array<T, Size<N>> {
    fn as_any(&self) -> Any {
        self.any
    }

    fn ty() -> shame_graph::Ty {
        let array_kind = shame_graph::Array(Rc::new(T::ty()),  Some(N));
        shame_graph::Ty::new(shame_graph::TyKind::Array(array_kind))
    }
    
    fn from_downcast(any: Any, stage: Stage) -> Self {
        Self { any, stage, last_mut_access: None, phantom: PhantomData }
    }
}

impl<T: Rec> Rec for Array<T, Unsized> {

    fn as_any(&self) -> Any {
        self.any
    }

    fn ty() -> shame_graph::Ty {
        let array_kind = shame_graph::Array(Rc::new(T::ty()),  None);
        shame_graph::Ty::new(shame_graph::TyKind::Array(array_kind))
    }
    
    fn from_downcast(any: Any, stage: Stage) -> Self {
        Self { any, stage, last_mut_access: None, phantom: PhantomData }
    }
}

impl<T: Rec, const N: usize> IntoRec for Array<T, Size<N>> {
    type Rec = Self;

    fn rec(self) -> Self {self}
    fn into_any(self) -> Any {self.any}
    fn stage(&self) -> Stage {self.stage}
}

impl<T: Rec> IntoRec for Array<T, Unsized> {
    type Rec = Self;

    fn rec(self) -> Self {self}
    fn into_any(self) -> Any {self.any}
    fn stage(&self) -> Stage {self.stage}
}

impl<T: Rec, Len: Sizing> Fields for Array<T, Len> where Self: Rec {
    
    fn parent_type_name() -> Option<&'static str> {None}

    fn from_fields_downcast(name: Option<&'static str>, f: &mut impl FnMut(shame_graph::Ty, &'static str) -> (Any, Stage)) -> Self {
        let (any, stage) = f(Self::ty(), name.unwrap_or("array"));
        Self::from_downcast(any, stage)
    }

    fn collect_fields(&self) -> Vec<(Any, Stage)> {
        vec![(self.any, self.stage)]
    }
}

impl<T: IntoRec, const N: usize> IntoRec for [T; N] {
    type Rec = Array<T::Rec, Size<N>>;

    fn rec(self) -> Self::Rec {Array::new(self)}
    fn into_any(self) -> Any {Array::new(self).any}
    fn stage(&self) -> Stage {self.iter().fold(Stage::Uniform, |acc, x| acc & x.stage())}
}

impl<T: Rec + Default + Copy, const N: usize> Default for Array<T, Size<N>>
where T: IntoRec<Rec=T> //always the case
{
    fn default() -> Self {
        //TODO: this might be a little wasteful wrt. graph nodes for large N
        [T::default(); N].rec()
    }
}

impl<T: Rec> Array<T, RecordedSize> {

    /// initialize a new [`Array<T, RecordedSize>`] array with `elements`
    pub fn new_with_size_of<X: IntoRec<Rec=T> + Clone>(elements: &[X]) -> Self {
        let args = elements.iter().map(|x| x.clone().rec());
        let stage = narrow_stages_or_push_error(args.clone().map(|x| x.stage()));
        let arg_anys = args.clone().map(|a| a.as_any()).collect::<Vec<_>>();
        let len = arg_anys.len();

        let array_ty = shame_graph::Array::new_sized(T::ty(), len);
        let any = Any::array_initializer(array_ty, &arg_anys);

        Array { 
            any,
            stage,
            last_mut_access: None,
            phantom: PhantomData,
        }
    }

}

impl<T: Rec, const N: usize> Array<T, Size<N>> {

    /// initialize a new [`Array<T, Size<N>>`] array with `elements`.
    /// 
    /// returns `None` if `elements.len() != N`
    pub fn try_new<X: IntoRec<Rec=T> + Clone>(elements: &[X]) -> Option<Self> {
        (elements.len() == N).then(|| {
            let args = elements.iter().map(|x| x.clone().rec());
            let stage = narrow_stages_or_push_error(args.clone().map(|x| x.stage()));
            let arg_anys = args.map(|a| a.as_any()).collect::<Vec<_>>();
    
            let array_ty = shame_graph::Array::new_sized(T::ty(), N);
            let any = Any::array_initializer(array_ty, &arg_anys);
            
            Array { 
                any,
                stage,
                last_mut_access: None,
                phantom: PhantomData,
            }
        })
    }

}