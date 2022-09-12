//! Traits that are implemented by recording types and types that can be turned
//! into recording types.
use super::{*};
use shame_graph::{Any, Ty};

/// Implemented by "recording types" such as `Array<T>` for arrays,
/// `Struct<T>` for structs, `Ten<S, D>` for tensors (which have aliases such
/// as `float4`, `int3x3`...)
///
/// Recording types are types that do not actually contain their data, but
/// instead point to a node in the shader's expression graph. Operations
/// performed on these types add further expression nodes to the expression
/// graph, no actual computation is performed. The resulting final graph is
/// traversed at the end of the shader recording to obtain the complete shader
/// code.
pub trait Rec: IntoRec + Sized {
    /// type erased `Any` object that exposes the full recording api
    /// (even functions that are not valid on its dynamic type)
    fn as_any(&self) -> Any;
    /// a struct describing the type `Self`, which is used by the underlying
    /// expression graph
    fn ty() -> Ty;
    /// downcast the provided type erased `any` value on a given `stage` to
    /// `Self`. This performs a runtime type check and panics if the runtime
    /// type does not correspond to `Self`.
    fn from_downcast(any: Any, stage: Stage) -> Self;

    /// set a preferred string for the identifier of this value in the generated
    /// shader code. (Sometimes the name will be modified slightly to avoid
    /// collisions with keywords, builtin functions and other identifiers)
    ///
    /// you are free to pass any name into this function, it should not cause
    /// a shader compiler error.
    fn aka(self, name: &str) -> Self {
        let stage = self.stage();
        Self::from_downcast(self.into_any().aka(name), stage)
    }
}

/// A trait implemented by recording types (`impl Rec`) and types that can be
/// converted into recording types e.g. rust literals, tuples, structs, arrays.
///
/// This trait mainly provides the `.rec()` method which can be used for
/// convenient creation of recording types
/// ```text
/// // create a float3 recording object
/// let my_vector = (1.0, 2.0, 3.0).rec();
///
/// // create an Array<float3, Size<5>> recording object
/// let my_array = [my_vector; 5].rec();
///
/// // create a Struct<A> recording object
/// #[derive(Fields)] struct A {member: float3}
/// let my_struct: Struct<A> = A {
///     member: my_vector
/// }.rec();
/// ```
/// for matrices, use the `MatCtor` trait, which provides `.mat_cols()` and `.mat_rows()`
/// to create matrices by interpreting vectors as columns or rows respectively.
///
pub trait IntoRec {
    /// `Self` when turned into a recording type (i.e. `shame::float` for `f32`)
    type Rec: Rec;

    /// turns `self` into a recording type `Rec`, which represents a node in the
    /// expression graph. Operations performed on the recording type object will
    /// push operations onto the expression graph.
    /// After the entire shader recording, the graph is traversed to generate a
    /// shader.
    fn rec(self) -> Self::Rec;

    /// gets the type erased `Any` recording type by either turning `Self` into
    /// a [`Rec`] type and then getting its contained `any` object, or getting
    /// the any object directly if `Self` is already [`Rec`]
    fn into_any(self) -> Any;

    /// returns the stage of `Self` or the stage that `Self` would have if
    /// turned into a [`Rec`] type
    fn stage(&self) -> Stage;
}