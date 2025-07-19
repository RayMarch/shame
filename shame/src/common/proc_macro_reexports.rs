//! functions/types/traits used by the proc-macro generated code

pub use super::proc_macro_utils::*;

pub use crate::call_info;
pub use crate::frontend::any::render_io::VertexAttribFormat;
pub use crate::frontend::any::shared_io::BindPath;
pub use crate::frontend::any::shared_io::BindingType;
pub use crate::frontend::any::Any;
pub use crate::frontend::any::InvalidReason;
pub use crate::frontend::encoding::buffer::BufferAddressSpace;
pub use crate::frontend::encoding::buffer::BufferInner;
pub use crate::frontend::encoding::buffer::BufferRefInner;
pub use crate::frontend::rust_types::layout_traits::ArrayElementsUnsizedError;
pub use crate::frontend::rust_types::layout_traits::CpuAligned;
pub use crate::frontend::rust_types::layout_traits::CpuLayout;
pub use crate::frontend::rust_types::layout_traits::FromAnys;
pub use crate::frontend::rust_types::layout_traits::GetAllFields;
pub use crate::frontend::rust_types::layout_traits::GpuLayout;
pub use crate::frontend::rust_types::layout_traits::VertexLayout;
pub use crate::frontend::rust_types::reference::AccessMode;
pub use crate::frontend::rust_types::reference::AccessModeReadable;
pub use crate::frontend::rust_types::reference::Ref;
pub use crate::frontend::rust_types::struct_::BufferFields;
pub use crate::frontend::rust_types::struct_::SizedFields;
pub use crate::frontend::rust_types::type_layout::FieldLayout;
pub use crate::frontend::rust_types::type_layout::FieldLayoutWithOffset;
pub use crate::frontend::rust_types::type_layout::layoutable::FieldOptions;
pub use crate::frontend::rust_types::type_layout::StructLayout;
pub use crate::frontend::rust_types::type_layout::Repr;
pub use crate::frontend::rust_types::type_layout::repr;
pub use crate::frontend::rust_types::type_layout::TypeLayout;
pub use crate::frontend::rust_types::type_layout::TypeLayoutSemantics;
pub use crate::frontend::rust_types::type_traits::BindingArgs;
pub use crate::frontend::rust_types::type_traits::GpuAligned;
pub use crate::frontend::rust_types::type_traits::GpuSized;
pub use crate::frontend::rust_types::type_traits::GpuStore;
pub use crate::frontend::rust_types::type_traits::GpuStoreImplCategory;
pub use crate::frontend::rust_types::type_traits::NoAtomics;
pub use crate::frontend::rust_types::type_traits::NoBools;
pub use crate::frontend::rust_types::type_traits::NoHandles;
pub use crate::frontend::rust_types::type_traits::VertexAttribute;
pub use crate::frontend::rust_types::type_traits::GpuLayoutField;
pub use crate::frontend::rust_types::type_layout::layoutable::SizedStruct;
pub use crate::frontend::rust_types::type_layout::layoutable::LayoutableType;
pub use crate::frontend::rust_types::type_layout::layoutable::SizedType;
pub use crate::frontend::rust_types::type_layout::layoutable::SizedOrArray;
pub use crate::frontend::rust_types::type_layout::layoutable::builder::StructFromPartsError;
pub use crate::frontend::rust_types::AsAny;
pub use crate::frontend::rust_types::GpuType;
#[allow(missing_docs)]
pub mod ir {
    pub use crate::ir::*;
}
pub use crate::frontend::any;
pub use crate::ir::ir_type::BufferBlock;
pub use crate::ir::ir_type::BufferBlockDefinitionError;
pub use crate::ir::pipeline::StageMask;
pub use crate::ir::recording::CallInfo;
pub use crate::ir::recording::CallInfoScope;
pub use crate::mem::AddressSpace;
pub use crate::any::U32PowerOf2;
