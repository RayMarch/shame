
use std::ops::{Deref, Range};

use crate::{any::Any, common::{IteratorExt, ranges_overlap}, context::Context, error::Error};

use super::*;

pub type Loc = u32; //Location, Set index or Binding index

pub struct InterfaceBlock(pub(crate) Vec<Any>);

impl Deref for InterfaceBlock {
    type Target = [Any];
    fn deref(&self) -> &[Any] {&self.0}
}

impl InterfaceBlock {
    pub fn new(members: Vec<Any>) -> Self {
        Self(members)
    }
}

pub enum Binding {
    Opaque (OpaqueTy, Any),
    OpaqueImage {
        tex_dtype_dims: TexDtypeDimensionality,
        any: Any,
        is_read_only: bool,
        tex_format: (), //TODO: add texture format enum in shame_graph
    },
    UniformBlock(InterfaceBlock),
    StorageMut  (InterfaceBlock),
    Storage     (InterfaceBlock),
}

pub struct BindGroup(pub(crate) Vec<(Loc, Binding)>);

impl BindGroup {
    pub fn new(bindings: Vec<(Loc, Binding)>) -> Self {
        Self(bindings)
    }
}

impl BindGroup {
    pub fn bindings(&self) -> &[(Loc, Binding)] {
        &self.0
    }
}

#[derive(Default)]
pub struct SideEffects {
    pub(crate) bind_groups: Vec<(Loc, BindGroup)>,
    pub(crate) push_constant: Option<Any>,
}

impl SideEffects {
    pub fn bind_groups(&self) -> &[(Loc, BindGroup)] {
        &self.bind_groups
    }
}

#[derive(Default)]
pub struct PrimitiveAssembly {
    pub(crate) vertex_attributes: Vec<(Range<Loc>, Any)>,
}

impl SideEffects {
    pub fn push_bind_group(&mut self, bind_group_index: Loc, bind_group: BindGroup) {
        self.bind_groups.push((bind_group_index, bind_group));
    }

    pub fn set_push_constant(&mut self, tensor: Tensor, ident: Option<String>) -> Any {
        let any = Any::global_interface(tensor.into_ty(), ident);
        assert!(self.push_constant.is_none(), "only one push constant can be defined per shader");
        self.push_constant = Some(any);
        any
    }
}

impl PrimitiveAssembly {
    pub fn push_vertex_attribute(&mut self, range: Range<Loc>, tensor: Tensor, ident: Option<String>) -> Any {
        assert!(range.len() == tensor.shape.col_count(), "range {:?} width does not match the amount of columns in {:?}", range, tensor.shape);
        let any = Any::global_interface(tensor.into_ty().as_const(), ident);
        self.vertex_attributes.push((range, any));
        any
    }

    /// returns a pair of the vertex attribute's recording type and its location range
    pub fn push_vertex_attribute_with_location_iter<T: Iterator<Item=u32>>(&mut self, loc_iter: &mut LocationIter<T>, tensor: Tensor, ident: Option<String>) -> (Any, Range<u32>) {
        let loc_range = loc_iter.next(tensor.shape.col_count() as u32)
        .expect("vertex attribute iterator exhausted"); //TODO: think about making this function return a Result, and all the repercussions of that
        let any = Any::global_interface(tensor.into_ty().as_const(), ident);
        self.vertex_attributes.push((loc_range.clone(), any));
        (any, loc_range)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Interpolation {
    Flat, //glsl `flat`
    PerspectiveLinear, //glsl `smooth`
    Linear, //glsl `noperspective`
}

#[derive(Debug, PartialEq, Eq)]
pub enum InOut {
    In,
    Out,
}

//for vertex shader outputs and fragment shader inputs
pub struct Varyings(pub(crate) InOut, pub(crate) Vec<(Interpolation, Any)>);

#[derive(Default)]
pub struct Framebuffer {
    pub(crate) color_attachments: Vec<(Loc, Any)>,
}

impl Framebuffer {
    pub fn push_color_attachment(&mut self, location: Loc, tensor: Tensor, ident: Option<String>) -> Any {
        assert!(tensor.shape.is_scalar_or_vec(), "color attachment tensor types may only be scalar or vec");
        let any = Any::global_interface(tensor.into_ty().as_write_only(), ident);
        self.color_attachments.push((location, any));
        any
    }
}

pub enum StageInterface {
    Vertex {
        inputs: PrimitiveAssembly,
        outputs: Varyings,
    },
    Fragment {
        inputs: Varyings,
        outputs: Framebuffer,
    },
    Compute {
        workgroup_size: Option<[usize; 3]>,
    },
}

impl Varyings {

    pub fn new(in_out: InOut) -> Self {
        Self(in_out, Default::default())
    }

    pub fn push_interpolated(&mut self, terp: Interpolation, tensor: Tensor, ident: Option<String>) -> (Any, Loc) {
        let Varyings(in_out, anys) = self;
        let access = match in_out {
            InOut::In  => Access::CopyOnWrite, //TODO: this might have to become a CopyOnWrite?
            InOut::Out => Access::WriteOnly,
        };
        let any = Any::global_interface(tensor.into_ty().into_access(access), ident);
        let loc = anys.len();
        anys.push((terp, any));
        (any, loc as Loc)
    }

    pub fn validate(&self, name_in_errors: &str) -> Result<(), Error> {
        use DType::*;

        Context::with(|ctx| self.1.iter().try_for_each(|(terp, any)| {

            match any.ty(&ctx.exprs()) {
                Some(ty) => match ty.kind {
                    TyKind::Tensor(Tensor { dtype, shape:_ }) => match dtype {
                        Bool => Err(Error::TypeError(format!("Interpolating `{ty}` is invalid. {name_in_errors} cannot be of any bool type"))),
                        F64 | I32 | U32 if terp != &Interpolation::Flat => {
                            Err(Error::TypeError(format!("{name_in_errors} {ty} must use Flat interpolation, but uses: {:?}", terp)))
                        }
                        _ => Ok(())
                    },
                    _ => Err(Error::TypeError(format!("invalid {name_in_errors} type: {ty}"))),
                }
                None => Ok(()), //invalid anys will be ignored
            }

        }))

    }
}

impl StageInterface {
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            StageInterface::Vertex { inputs, outputs } => {
                if let Some((a, b)) = inputs.vertex_attributes.iter().find_pair(|(a, _), (b, _)| {
                    ranges_overlap(a, b).then(|| (a, b))
                }) {
                    Err(Error::OverlappingAttributeLocation(a.clone(), b.clone()))?;
                }

                outputs.validate("vertex shader output")?;
            },
            StageInterface::Fragment { inputs, outputs } => {
                inputs.validate("fragment input")?;

                if let Some(loc) = outputs.color_attachments.iter().find_pair(|(a, _), (b, _)| (a == b).then(|| a)) {
                    Err(Error::OverlappingColorAttachmentLocation {
                        duplicate_location: *loc,
                    })?;
                }
            },
            StageInterface::Compute {..} => (),
        }
        Ok(())
    }
}

impl SideEffects {
    pub fn validate(&self) -> Result<(), Error> {
        if let Some(overlap) = self.bind_groups.iter().find_pair(|(a, _), (b, _)| {
            (a == b).then(|| a)
        }) {
            Err(Error::OverlappingBindGroupIndex{duplicate_index: *overlap})?;
        }

        for (group_i, group) in &self.bind_groups {

            if let Some(overlap) = group.0.iter().find_pair(|(a, _), (b, _)| {
                (a == b).then(|| a)
            }) {
                Err(Error::OverlappingBindingIndex {
                    bind_group: *group_i,
                    duplicate_index: *overlap
                })?;
            }

            for (_, binding) in &group.0 {
                match binding {
                    Binding::Opaque(opaque_ty, any) => match opaque_ty {
                        OpaqueTy::TextureCombinedSampler(_) |
                        OpaqueTy::ShadowSampler(_) => {
                            Context::with(|ctx| -> Result<(), Error> {
                                match any.ty_via_ctx(ctx) {
                                    Some(ty) if ty.access != Access::Const => {
                                        Err(Error::TypeError(format!("sampler types like '{}' must be const", ty)))
                                    },
                                    _ => Ok(()),
                                }
                            })?;
                        },
                        _ => (),
                    },
                    Binding::Storage(interface_block) |
                    Binding::StorageMut(interface_block)
                    => match interface_block.split_last() {
                        Some((_last, rest)) => {
                            let result = Context::with(|ctx| {
                                let exprs = ctx.exprs();
                                let rest_all_sized = rest.iter().all(|any| {
                                    any.ty(&exprs).map(|ty| ty.is_sized()).unwrap_or(true)
                                });
                                //whether last is sized or not does not matter.
                                if !rest_all_sized {
                                    Err(Error::TypeError(
                                        "storage block contains unsized type at a field that is not the last one. Only the last field in a storage block can be of unsized type"
                                        .to_string()
                                    ))
                                } else {
                                    Ok(())
                                }
                            });
                            if let Err(e) = result {
                                return Err(e)
                            }
                        },
                        None => (),
                    },
                    Binding::UniformBlock(interface_block) => {
                        let all_sized = interface_block.iter().all(|x| x.ty_via_thread_ctx().map(|ty| ty.is_sized()).unwrap_or(true));
                        if !all_sized {
                            return Err(Error::TypeError(
                                "uniform block cannot contain unsized types"
                                .to_string()
                            ))
                        };
                    }
                    _ => (),
                }
            }
        }
        Ok(())
    }
}

impl Shader {
    pub fn validate_interface(&self) -> Result<(), Error> {
        self.side_effects.validate()?;
        self.stage_interface.validate()?;
        Ok(())
    }
}

// some examples for the higher layers on top of this library
// struct Uniform   <T: UniformBlock>(T); //impl Bindable for Uniform
// struct Storage   <T: UniformBlock>(T); //impl Bindable for Uniform
// struct StorageMut<T: UniformBlock>(T); //impl Bindable for Uniform

pub trait UniformBlock { //derive(UniformBlock)
    type RecordingType;
    fn get_recording_type(_: &SideEffects, access: Access) -> Self::RecordingType;
    fn as_struct_layout(&self, _: &SideEffects) -> InterfaceBlock;
}

pub trait Bindable { //each of a derive(BindGroup) type's members
    type RecordingType;
    fn get_recording_type(_: &SideEffects) -> Self::RecordingType;
    fn as_binding(&self, _: &SideEffects) -> Binding; //calls as_struct_layout of inner T for Storage<T> etc
}

pub trait Group {
    fn new_for_shader(_: &mut SideEffects) -> Self;
}
