//! record-time type system.
//!
//! This is the type system that is used during the [`ExecutionPhase::RecordTime`]

mod align_size;
mod canon_name;
mod categories;
mod memory_view;
mod struct_;
mod tensor;
mod texture;
mod texture_new;
mod ty;

pub use align_size::*;
pub use canon_name::*;
pub use memory_view::*;
pub use struct_::*;
pub use tensor::*;
pub use texture_new::*;
pub use ty::*;
