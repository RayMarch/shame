//! recording types
//! 
//! types and functions that record the actions performed on them into the
//! shame_graph, so that a shader can be generated later.

pub mod control_flow;

pub mod fields;

pub mod rec;
pub use rec::*;

pub mod array;
pub use array::*;

pub mod struct_;
pub use struct_::*;

pub mod ten;
pub use ten::*;

pub mod stage;
pub use stage::*;

pub mod aliases;
pub use aliases::*;

pub mod swizzle;
pub use swizzle::*;

pub mod constructors;
pub use constructors::*;

pub mod shape;
pub use shape::*;

pub mod dtype;
pub use dtype::*;

pub mod operators;
pub use operators::*;

pub mod write_only;
pub use write_only::*;

pub mod functions;
pub use functions::*;

pub mod texture_combined_sampler;
pub use texture_combined_sampler::*;

pub mod multi;
pub use multi::*;

pub mod sampler_texture;