//! functions for creation of structs in shaders, [`Struct<T: Fields>`]
use std::ops::{Deref, DerefMut};
use shame_graph::{Any, Ty, TyKind, Named};
use super::{Stage, Rec, IntoRec, fields::Fields, narrow_stages_or_push_error};

#[derive(Clone, Copy)]
// FIXME: the T within structs can be accessed mutably, which is fine for 
// operators like `+=` etc, but might result in unexpected behavior when using `=`
// to overwrite the member recording-reference (instead of overwriting its value). 
// This can be made less surprising if T is RefCell'd and recreated/validated on 
// Deref::deref and Deref::deref_mut. That way a "wrong" later access after modification 
// of members can be detected and an error can be pushed
/// A struct [`Rec`] recording type that uses `T`'s fields as the struct's 
/// fields.
pub struct Struct<T: Fields> {
    t: T,
    any: Any,
    stage: Stage,
}

impl<T: Fields> Rec for Struct<T> {
    fn as_any(&self) -> Any {self.any}
    fn ty() -> Ty {Ty::new(TyKind::Struct(Self::get_or_declare_struct()))} //the derived `T::struct_ty()` declares the struct if needed

    fn from_downcast(any: Any, stage: Stage) -> Self {
        let struct_ = Self::get_or_declare_struct();
        let t = prepare_field_selects::<T>((any, stage), struct_);
        Struct {t, any, stage}
    }
}

impl<T: Fields> IntoRec for Struct<T> {
    type Rec = Self;

    fn rec(self) -> Self::Rec {self}
    fn into_any(self) -> Any {self.any}
    fn stage(&self) -> Stage {self.stage}
}

impl<T: Fields> Struct<T> {

    /// create a new [`Struct<T>`] with the fields of `T` initialized to
    /// `initial_values`
    pub fn new(initial_values: T) -> Self {

        let fields: Vec<(Any, Stage)> = initial_values.collect_fields();

        let anys   = fields.iter().map(|(any, _)| *any).collect::<Vec<_>>();
        let stages = fields.iter().map(|(_, stage)| *stage);

        let any = Any::struct_initializer(
            Self::get_or_declare_struct(), 
            &anys
        );

        let stage = narrow_stages_or_push_error(stages);
        Struct::<T>::from_downcast(any, stage)
    }

    /// assignment to `self`. Use this instead of the `=` operator for assigning to
    /// the underlying struct value in the shader
    pub fn set(&mut self, val: &Self) {
        //decided to take a ref as first argument because Self might not impl copy, and a clone() by the user is unnecessary
        narrow_stages_or_push_error([self.stage, val.stage]);
        self.any.assign(val.any)
    }

    /// called internally by the generated code in `derive(shame::Fields)`
    /// registers a new struct type.
    pub fn get_or_declare_struct() -> shame_graph::Struct {
        let mut fields = Vec::new();
        T::for_each_field(|ty, field_name| fields.push((field_name, ty)));
        shame_graph::Context::with(|ctx| {
            ctx.get_or_insert_struct(T::parent_type_name().unwrap_or("AnonymousStruct"), &fields)
        })
    }

    /// records creation of a copy of this struct. 
    /// 
    /// Not to be confused with the behavior of the `Clone` or `Copy` traits, which the
    /// user may define on their `T: Fields` (and thus derive on `Struct<T>` aka `Self`), which
    /// would merely clone the recording reference and have no effect on the resulting shader
    pub fn copy(&self) -> Self {
        let any = self.any.copy();
        let t = prepare_field_selects((any, self.stage), Self::get_or_declare_struct());
        Self{t, any, stage: self.stage}
    }
}

impl<T: Fields> Fields for Struct<T> {
    fn parent_type_name() -> Option<&'static str> {
        None
    }

    fn from_fields_downcast(name: Option<&'static str>, f: &mut impl FnMut(shame_graph::Ty, &'static str) -> (Any, Stage)) -> Self {
        let (any, stage) = f(Self::ty(), name.unwrap_or("struct"));
        Self::from_downcast(any, stage)
    }

    fn collect_fields(&self) -> Vec<(Any, Stage)> {
        vec![(self.any, self.stage)]
    }
}

/// fills all fields in `T` with a struct `field_select` expression so that the user can write `t.field` and it will contain the expected `Any`
pub fn prepare_field_selects<T: Fields>(parent: (Any, Stage), s: shame_graph::Struct) -> T {

    let (p_any, p_stage) = parent;

    let shame_graph::Struct(Named(fields, _ident)) = &s;
    let mut idents = fields.iter().map(|Named(_, ident)| *ident).fuse();

    let mut num_assigned_fields = 0;

    let t = T::from_fields_downcast(None, &mut |_, _| {
        num_assigned_fields += 1;

        let any = match idents.next() {
            Some(x) => p_any.field_select(x),
            None => Any::not_available(),
        };
        
        (any, p_stage)
    });

    if num_assigned_fields != fields.len() {
        shame_graph::Context::with(|ctx| {
            ctx.push_error(shame_graph::Error::AssertionFailed(
                format!("failed to fill rust struct with field accesses. {} type expected {} fields but got {}.", s, fields.len(), num_assigned_fields)
            ))
        })
    }
    
    t
}

impl<T: Fields> Deref for Struct<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {&self.t}
}

impl<T: Fields> DerefMut for Struct<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {&mut self.t}
}

impl<T: Fields + Default> Default for Struct<T> 
where T: IntoRec<Rec=Self> //always the case 
{
    fn default() -> Self {
        T::default().rec()
    }
}