
use std::{cell::{Cell, RefCell}, rc::Rc};
use crate::{context::ShaderKind, pool::Key};
use super::*;

pub enum Item {
    //args have internal mutability because they are discovered one after another by usage across item-boundary
    FuncDef {ident: IdentSlot, args: Rc<RefCell<Vec<Named<Ty>>>>, body: Cell<Option<Key<Block>>>}, //body is None while recording is in progress
    MainFuncDef {body: Cell<Option<Key<Block>>>}, //body is None while recording is in progress
    StructDef(Struct),
}

pub struct Shader {
    pub side_effects: SideEffects, //bindgroups, uniforms
    pub stage_interface: StageInterface, //vertex attributes, rendertargets
}

impl Shader {
    pub fn new(shader_kind: ShaderKind) -> Self {
        Self {
            side_effects: Default::default(),
            stage_interface: match shader_kind {
                ShaderKind::Vertex => StageInterface::Vertex {
                    inputs : Default::default(),
                    outputs: Varyings::new(InOut::Out),
                },
                ShaderKind::Fragment => StageInterface::Fragment {
                    inputs : Varyings::new(InOut::In),
                    outputs: Default::default(),
                },
                ShaderKind::Compute => StageInterface::Compute {
                    workgroup_size: None,
                },
            },
        }
    }
}