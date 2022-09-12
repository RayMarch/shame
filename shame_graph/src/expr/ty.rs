
use std::{fmt::Display, rc::Rc};
use crate::{Context, Error};

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Struct(pub Named<Rc<Vec<Named<Ty>>>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Callable(Named<(Rc<Ty>, Rc<Vec<Ty>>)>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexDtypeDimensionality(pub DType, pub TexDimensionality);

impl TexDtypeDimensionality {
    pub fn new(dtype: DType, kind: TexDimensionality) -> Self {
        assert!([DType::I32, DType::U32, DType::F32].contains(&dtype), "glsl only supports i32, u32 or f32 sampler types");
        Self(dtype, kind)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpaqueTy {
    TextureCombinedSampler(TexDtypeDimensionality), //glsl samplerND, see glsl spec 5.4.5. Texture-Combined Sampler Constructors
    ShadowSampler(ShadowSamplerKind),
    Sampler, //glsl sampler, wgpu bindingtype sampler
    Texture(TexDtypeDimensionality), //glsl textureND, wgpu bindingtype texture
    Image(TexDtypeDimensionality), //not implemented yet //glsl image, wgpu bindingtype storage texture
    AtomicCounter(()), //Not Implemented Yet
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array(pub Rc<Ty>, pub Option<usize>);
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayOfOpaque(pub(crate) OpaqueTy, pub(crate) Option<usize>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyKind {
    Void,

    Tensor(Tensor),
    Struct(Struct),
    Callable(Callable),
    Array(Array),

    Opaque(OpaqueTy),
    ArrayOfOpaque(ArrayOfOpaque),

    InterfaceBlock(Struct),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    LValue,
    CopyOnWrite, //most values are this, which means that if the value is written to in a shader recording, the value will get assigned to a variable first (copying it) //TODO: a lot has happened since these names were given to access types. Review whether this is still accuarte, if not, choose a better name (also see comment in Array::at_mut)
    Const,
    WriteOnly,
}

impl Default for Access {
    fn default() -> Self {Self::CopyOnWrite}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ty {
    pub kind: TyKind,
    pub access: Access,
}

macro_rules! define_ty_tensor_init {
    ($($name: ident),*) => {
        $(pub const fn $name() -> Self {Self{access: Access::CopyOnWrite, kind: TyKind::Tensor(Tensor::$name())}})*
    }
}

impl Ty {
    pub fn new(kind: TyKind) -> Self {Self {
        kind,
        access: Access::default(),
    }}

    pub fn new_access(kind: TyKind, access: Access) -> Self {Self {
        kind,
        access,
    }}

    pub fn tensor(shape: Shape, dtype: DType) -> Self {
        Self {
            access: Access::default(),
            kind: TyKind::Tensor(Tensor::new(shape, dtype)),
        }
    }

    pub fn array(item: Ty, size: Option<usize>) -> Ty {
        Self {
            kind: TyKind::Array(Array(Rc::new(item), size)),
            access: Access::default(),
        }
    }

    /// whether the type's byte size is known at shader compile time
    pub fn is_sized(&self) -> bool {
        // this is even considering invalid types, such as Array<UnsizedType>
        match &self.kind {
            TyKind::Tensor(_) => true,
            TyKind::Struct(Struct(Named(fields, _))) => {
                fields.iter().find(|Named(ty, _)| !ty.is_sized()).is_none()
            },
            TyKind::Array(Array(ty, size)) => ty.is_sized() && size.is_some(),
            TyKind::Void |
            TyKind::Callable(_) |
            TyKind::Opaque(_) |
            TyKind::ArrayOfOpaque(_) |
            TyKind::InterfaceBlock(_) => true
        }
    }

    pub fn attribute_location_width(&self) -> usize {
        match self.kind {
            TyKind::Tensor(x) => x.shape.col_count(),
            TyKind::Struct(_) => todo!(),
            TyKind::Array(_) => todo!(),
            TyKind::InterfaceBlock(_) => todo!(),
            _ => panic!("unable to get attribute location width of type {:?}", self)
        }
    }

    pub fn texture_combined_sampler(dtype: DType, dims: TexDimensionality) -> Self {Self {
        kind: TyKind::Opaque(OpaqueTy::TextureCombinedSampler(TexDtypeDimensionality::new(dtype, dims))),
        access: Access::Const, //opaque types need to be const
    }}

    pub fn try_as_tensor(&self) -> Option<Tensor> {
        match self.kind {
            TyKind::Tensor(ten) => Some(ten),
            _ => None,
        }
    }

    pub fn void() -> Ty {
        Ty::new(TyKind::Void)
    }

    pub fn eq_ignore_access(&self, rhs: &Ty) -> bool {
        self.kind == rhs.kind
    }

    pub fn into_access(self, access: Access) -> Ty {Self::new_access(self.kind, access)}
    //TODO: turn as_const into "into_const" bc most of the time we can consume the type anyways
    pub fn as_const     (&self) -> Ty {Self {kind: self.kind.clone(), access: Access::Const      }}
    pub fn as_write_only(&self) -> Ty {Self {kind: self.kind.clone(), access: Access::WriteOnly  }}
    pub fn as_cow       (&self) -> Ty {Self {kind: self.kind.clone(), access: Access::CopyOnWrite}}
    pub fn as_lvalue    (&self) -> Ty {Self {kind: self.kind.clone(), access: Access::LValue     }}

    //calls the Tensor constructors with default mutability
    define_ty_tensor_init!(
        float, double,
        int, uint,
        bool,
        vec2, vec3, vec4,
        bvec2, bvec3, bvec4,
        dvec2, dvec3, dvec4,
        ivec2, ivec3, ivec4,
        uvec2, uvec3, uvec4,
        mat2  , mat2x3, mat2x4,
        mat3x2, mat3  , mat3x4,
        mat4x2, mat4x3, mat4,
        dmat2  , dmat2x3, dmat2x4,
        dmat3x2, dmat3  , dmat3x4,
        dmat4x2, dmat4x3, dmat4
    );

}

impl From<Tensor> for Ty {
    fn from(t: Tensor) -> Self {
        Self::new(TyKind::Tensor(t))
    }
}

impl Display for Access {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Access::LValue => "lvalue",
            Access::CopyOnWrite => "copy-on-write",
            Access::Const => "const",
            Access::WriteOnly => "write-only",
        })
    }
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.access.fmt(f)?;
        f.write_str(" ")?;
        self.kind.fmt(f)
    }
}

impl Display for TyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TyKind::Void => f.write_str("void"),
            TyKind::Tensor(x) => x.fmt(f),
            TyKind::Struct(x) => x.fmt(f),
            TyKind::Callable(_) => todo!("display callable"),
            TyKind::Array(x) => x.fmt(f),
            TyKind::Opaque(x) => x.fmt(f),
            TyKind::ArrayOfOpaque(_) => todo!("display array-of-opaque"),
            TyKind::InterfaceBlock(_) => todo!("display interface-block"),
        }
    }
}

impl Display for OpaqueTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpaqueTy::TextureCombinedSampler(x) => f.write_fmt(format_args!("TextureCombinedSampler{x}")),
            OpaqueTy::ShadowSampler(x) => x.fmt(f),
            OpaqueTy::Sampler => f.write_str("sampler"),
            OpaqueTy::Texture(x) => f.write_fmt(format_args!("Texture{x}")),
            OpaqueTy::Image(x) => f.write_fmt(format_args!("Image{x}")),
            OpaqueTy::AtomicCounter(_) => todo!(),
        }
    }
}

impl Display for TexDtypeDimensionality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let TexDtypeDimensionality(dtype, kind) = self;
        f.write_fmt(format_args!("{:?}<{}>", kind, dtype))
    }
}

impl Display for ShadowSamplerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("shadow_sampler{:?}", self))
    }
}

impl Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Array(elem_ty, maybe_len) = self;
        let elem_ty_kind = &elem_ty.kind;
        match maybe_len {
            Some(len) => f.write_fmt(format_args!("array<{elem_ty_kind}, {len}>")),
            None      => f.write_fmt(format_args!("array_unsized<{elem_ty_kind}>")),
        }
    }
}

impl Display for Struct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Struct(Named(fields, ident)) = self;
        Context::with(|ctx| {
            let idents = ctx.idents();

            let fields_string = fields.iter().map(|Named(ty, ident)| {

                let ident_string = idents[**ident].clone().unwrap_or_else(|| "_".to_string());
                format!("    {ident_string}: {ty},\n")

            })
            .collect::<String>();

            let struct_name = idents[**ident].to_owned().unwrap_or_else(|| "<anonymous>".to_string());

            f.write_fmt(format_args!("struct {struct_name} {{\n{fields_string}}}"))
        })
    }
}

impl Struct {

    pub fn from_name_fields(ctx: &crate::Context, name: &str, fields: &[(&str, Ty)]) -> Self {
        let mut idents = ctx.idents_mut();

        let name = IdentSlot::new_in(Some(name.to_string()), &mut idents);

        let fields = fields.iter().map(|(name, ty)| {
            if !ty.is_sized() {
                ctx.push_error(Error::TypeError(format!(
                    "type {} is unsized and cannot be field in a struct. Only sized types are allowed as struct fields.",
                    &ty
                )));
            }
            let name = IdentSlot::new_in(Some(name.to_string()), &mut idents);
            Named(ty.clone(), name)
        }).collect();

        Self(Named(Rc::new(fields), name))
    }

    /// test whether a struct corresponds to given name and field description without creating Pool `Key`s for them
    pub fn eq_name_fields(&self, name: &str, fields: &[(&str, Ty)]) -> bool {
        let &Struct(Named(fields_, name_)) = &self;

        name_.eq_str(name) &&
        fields.iter().zip(fields_.iter()).all(|((name, ty), Named(ty_, name_))| {
            name_.eq_str(name) &&
            ty_.eq_ignore_access(ty)
        })
    }

    pub fn find_field_ident(&self, field_name: &str) -> Option<IdentSlot> {
        let &Struct(Named(fields, _)) = &self;

        fields.iter().find_map(|Named(_, ident)|
            ident.eq_str(field_name).then(|| ident.clone())
        )
    }
}

impl Array {
    pub fn new(element_ty: Ty, maybe_len: Option<usize>) -> Self {
        Array(Rc::new(element_ty), maybe_len)
    }

    pub fn new_sized(element_ty: Ty, len: usize) -> Self {
        Self::new(element_ty, Some(len))
    }
}