//! the [`Fields`] struct which is derived by `#[derive(Fields)]`

use shame_graph::{Any, Ty};

use super::{IntoRec, Stage};

/// implemented on structs via `#[derive(Fields)]`.
/// This trait provides a way to make rust structs conveniently work with shader
/// recording types.
/// Deriving this type on `T` will make it possible for `T` to be used in
/// `Struct<T>` or shader inputs such as
/// - vertex attributes (tensors only)
/// - storage buffers
/// - uniform blocks
///
/// each of which have their own restrictions which will be checked at recording
/// time.
///
/// Ignore the functions of this trait, they are only used internally to
/// assemble the necessary information to create the shader inputs mentioned
/// above. Users should use the `RenderFeatures` or `ComputeFeatures` objects
/// provided in the shader recording to create those inputs.
pub trait Fields: IntoRec + Sized {
    /// name of the type containing the fields, returns None if Fields is
    /// implemented for a single recording type element
    fn parent_type_name() -> Option<&'static str>;

    /// `Self` constructor that visits each field and asks `f` to create a
    /// `(Any, Stage)` tuple for it
    fn from_fields_downcast(
        name: Option<&'static str>,
        f: &mut impl FnMut(Ty, &'static str) -> (Any, Stage),
    ) -> Self;

    /// collects the fields into a type erased vector
    fn collect_fields(&self) -> Vec<(Any, Stage)>;

    /// instantiate `Self` with its fields initialized as global interface
    /// variables
    fn new_as_interface_block(access: shame_graph::Access) -> (Self, shame_graph::InterfaceBlock) {
        let mut anys = vec![];
        let t = Self::from_fields_downcast(Self::parent_type_name(), &mut |ty, name| {
            let ty = ty.into_access(access);
            let any = Any::global_interface(ty, Some(name.to_string()));
            anys.push(any);
            (any, Stage::Uniform)
        });
        (t, shame_graph::InterfaceBlock::new(anys))
    }

    /// visit each field in `Self` with `f`
    fn for_each_field(mut f: impl FnMut(Ty, &'static str))
    where
        Self: Sized,
    {
        // implemented through `from_fields_downcast` in order to reduce the amount of things
        // that need to happen in the proc macro that derives this trait
        Self::from_fields_downcast(None, &mut |ty, field_name| {
            f(ty, field_name);
            (Any::not_available(), Stage::NotAvailable)
        });
    }
}
