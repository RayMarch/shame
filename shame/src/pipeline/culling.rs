//! face culling (clockwise/counterclockwise)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// specifies which primitives should be culled based on their winding order.
///
/// see <https://www.khronos.org/opengl/wiki/Face_Culling>.
///
/// to get the default behavior from common graphics APIs, use `Cull::default()` from the `Default` trait.
pub enum Cull {
    /// perform no primitive winding order based face culling. All primitives remain.
    Off,
    /// cull faces with counter-clockwise winding order, which are usually considered to be a "front face" by default
    CCW,
    /// cull faces with clockwise winding order, which are usually considered to be a "back face" by default
    ///
    /// alternatively just call `Cull::default()` from the `Default` trait
    CW,
}

impl Default for Cull {
    /// cull faces with clockwise winding order, which are usually considered to be a "back face" by default
    fn default() -> Self { Cull::CW }
}
