use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Display,
    ops::Deref,
    rc::Rc,
};

use thiserror::Error;

use super::{align_of_array, canon_name::CanonName, round_up, LayoutError, SizedType, StoreType, Type};
use crate::{
    call_info,
    common::{iterator_ext::IteratorExt, po2::U32PowerOf2, pool::Key},
    ir::recording::{Context, Ident},
};
use crate::{
    common::pool::PoolRefMut,
    ir::recording::{CallInfo, Priority},
};

#[allow(missing_docs)]
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum FieldDefinitionError {
    #[error("field `{0}` has a custom alignment of {1}. Alignment attributes must be a positive power of two")]
    CustomAlignNotAPowerOfTwo(CanonName, u64),
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum StructureDefinitionError {
    #[error("runtime sized arrays are only allowed as the last field of a structure definition")]
    /// required by https://www.w3.org/TR/WGSL/#struct-types
    RuntimeSizedArrayAtNonLastField,
    #[error("{0} definitions require at least one field")]
    /// required by https://www.w3.org/TR/WGSL/#struct-types
    MustHaveAtLeastOneField(StructKind),
    #[error("struct kind {0:?} does not match struct type `{1}`")]
    /// required by https://www.w3.org/TR/WGSL/#struct-types
    WrongStructKindForStructType(StructKind, &'static str),
    #[error(
        "runtime sized arrays are only allowed as the last field of a buffer-block struct. They are not allowed in sized structs."
    )]
    RuntimeSizedArrayNotAllowedInSizedStruct,
    #[error("field names must be unique within a structure definition")]
    FieldNamesMustBeUnique(StructureFieldNamesMustBeUnique),
}

pub trait Field {
    fn name(&self) -> &CanonName;
    fn byte_size(&self) -> Option<u64>;
    fn align(&self) -> u64;
    fn custom_min_size(&self) -> Option<u64>;
    fn custom_min_align(&self) -> Option<U32PowerOf2>;
    fn ty(&self) -> StoreType;
}

#[rustfmt::skip]
impl Field for SizedField {
    fn name(&self) -> &CanonName {&self.name}
    fn byte_size(&self) -> Option<u64> {Some(self.ty.byte_size().max(self.custom_min_size.unwrap_or(0)))}
    fn align(&self) -> u64 {self.ty.align().max(self.custom_min_align.map(u64::from).unwrap_or(1))}
    fn ty(&self) -> StoreType {StoreType::Sized(self.ty.clone())}
    fn custom_min_size(&self) -> Option<u64> {self.custom_min_size}
    fn custom_min_align(&self) -> Option<U32PowerOf2> {self.custom_min_align}
}

#[rustfmt::skip]
impl Field for RuntimeSizedArrayField {
    fn name(&self) -> &CanonName {&self.name}
    fn byte_size(&self) -> Option<u64> {None}
    fn align(&self) -> u64 {
        align_of_array(&self.element_ty).max(
            self.custom_min_align.map(u64::from).unwrap_or(1)
        )
    }
    fn custom_min_size(&self) -> Option<u64> {None}
    fn custom_min_align(&self) -> Option<U32PowerOf2> {self.custom_min_align}
    fn ty(&self) -> StoreType {StoreType::RuntimeSizedArray(self.element_ty.clone())}
}

#[allow(missing_docs)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizedField {
    pub name: CanonName,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
    pub ty: SizedType,
}

#[allow(missing_docs)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeSizedArrayField {
    pub name: CanonName,
    pub custom_min_align: Option<U32PowerOf2>,
    pub element_ty: SizedType,
}

#[allow(missing_docs)] // runtime api
impl SizedField {
    #[allow(missing_docs)] // runtime api
    pub fn new(
        name: CanonName,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
        ty: SizedType,
    ) -> Self {
        Self {
            name,
            custom_min_size,
            custom_min_align,
            ty,
        }
    }

    /// the alignment of this field, respecting user defined custom minimum alignment.
    pub fn align(&self) -> u64 { self.ty.align().max(self.custom_min_align.map(u64::from).unwrap_or(1)) }

    /// the byte-size of this field, respecting user defined custom minimum size.
    pub fn byte_size(&self) -> u64 { self.ty.byte_size().max(self.custom_min_size.unwrap_or(0)) }

    pub fn ty(&self) -> &SizedType { &self.ty }
}

impl RuntimeSizedArrayField {
    #[allow(missing_docs)] // runtime api
    pub fn new(name: CanonName, custom_min_align: Option<U32PowerOf2>, array_element_ty: SizedType) -> Self {
        Self {
            name,
            custom_min_align,
            element_ty: array_element_ty,
        }
    }

    /// the alignment of this field, respecting user defined custom alignment.
    pub fn align(&self) -> u64 {
        self.element_ty
            .align()
            .max(self.custom_min_align.map(u64::from).unwrap_or(1))
    }

    #[allow(missing_docs)] // runtime api
    pub fn element_ty(&self) -> &SizedType { &self.element_ty }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Struct {
    kind: StructKind,
    // in `shame`, two structs with the same name but different fields may exist.
    // The actual chosen Identifier for this struct may not match the canonical name.
    name: CanonName,
    sized_fields: Vec<SizedField>,
    last_unsized: Option<RuntimeSizedArrayField>,
}

impl Struct {
    #[track_caller]
    pub fn new_sized(
        kind: StructKind,
        name: CanonName,
        mut first_sized_fields: Vec<SizedField>,
        last_sized_field: SizedField,
    ) -> Result<Rc<Self>, StructureFieldNamesMustBeUnique> {
        first_sized_fields.push(last_sized_field);
        let sized_fields = first_sized_fields;
        assert!(!sized_fields.is_empty());
        check_for_duplicate_field_names(&sized_fields, None)?;
        let struct_ = Rc::new(Self {
            kind: StructKind::Sized,
            name,
            sized_fields,
            last_unsized: None,
        });
        try_register_struct(call_info!(), &struct_);
        Ok(struct_)
    }

    #[track_caller]
    pub fn new(
        kind: StructKind,
        name: CanonName,
        sized_fields: Vec<SizedField>,
        last_unsized: Option<RuntimeSizedArrayField>,
    ) -> Result<Rc<Self>, StructureDefinitionError> {
        use StructKind as SK;
        let num_fields = sized_fields.len() + last_unsized.iter().count();
        if num_fields < 1 {
            return Err(StructureDefinitionError::MustHaveAtLeastOneField(kind));
        }
        if kind == SK::Sized && last_unsized.is_some() {
            return Err(StructureDefinitionError::RuntimeSizedArrayNotAllowedInSizedStruct);
        }

        let struct_ = Rc::new(Self {
            kind,
            name,
            sized_fields,
            last_unsized,
        });

        // try register struct if we're currently in a pipeline encoding,
        // otherwise the registration will happen later with a less useful `call_info`
        Context::try_with(call_info!(), |ctx| {
            ctx.struct_registry_mut().register_mentioned_structs_recursively(
                &struct_,
                &mut ctx.pool_mut(),
                ctx.latest_user_caller(),
            );
        });
        if let Err(e) = check_for_duplicate_field_names(&struct_.sized_fields, struct_.last_unsized.as_ref()) {
            return Err(StructureDefinitionError::FieldNamesMustBeUnique(e));
        }

        Ok(struct_)
    }

    pub(crate) fn map_sized_field_types(mut self, mut f: impl FnMut(SizedType) -> SizedType) -> Self {
        for field in self.sized_fields.iter_mut() {
            field.ty = f(field.ty.clone());
        }
        self
    }

    pub(crate) fn find_sized_field(&self, canonical_name: &CanonName) -> Option<&SizedField> {
        self.sized_fields.iter().find(|f| &f.name == canonical_name)
    }

    pub(crate) fn find_field(&self, canonical_name: &CanonName) -> Option<&dyn Field> {
        self.sized_fields
            .iter()
            .find(|f| &f.name == canonical_name)
            .map(|t| t as &dyn Field)
            .or_else(|| {
                self.last_unsized
                    .iter()
                    .find_map(|last| (last.name() == canonical_name).then_some(last as &dyn Field))
            })
    }

    /// the minimal byte size of this struct
    ///
    /// if there is no runtime sized array last member, returns the actual size.
    /// If there is a runtime sized array last member, returns the byte size this
    /// structure would have if there was one element in that array.
    ///
    pub fn min_byte_size(&self) -> u64 {
        // wgsl spec:
        //   roundUp(AlignOf(S), justPastLastMember)
        //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)
        let last_field_index = self.len() - 1;
        let last_field_min_byte_size = match &self.last_unsized {
            Some(last) => round_up(last.align(), last.element_ty.byte_size()),
            None => self.sized_fields[last_field_index].byte_size(),
        };

        let just_past_last = self.offset_of_member(last_field_index) + last_field_min_byte_size;
        round_up(self.align(), just_past_last)
    }

    /// the alignment in bytes
    pub fn align(&self) -> u64 {
        let sized_max: u64 = self.sized_fields.iter().map(|x| x.align()).max().unwrap_or(1);
        match &self.last_unsized {
            Some(last) => sized_max.max(last.align()),
            None => sized_max,
        }
    }

    /// the offset of the i’th member from the start of the host-shareable
    /// structure s.
    ///
    /// panics if i >= num_fields
    ///
    fn offset_of_member(&self, i: usize) -> u64 {
        assert!(i < self.len(), "struct {self:?} has no {i}'th member");

        let mut curr_offset = 0;
        for curr_i in 0..i {
            let next_i = curr_i + 1;

            let curr = &self.sized_fields[curr_i];
            let align_of_next = match next_i {
                x if x < self.sized_fields.len() => self.sized_fields[next_i].align(),
                x => match &self.last_unsized {
                    Some(x) => x.align(),
                    None => unreachable!("num_fields was calculated to include the optional last field"),
                },
            };

            // WGSL spec:
            // OffsetOfMember(S, i) = roundUp(AlignOfMember(S, i ), OffsetOfMember(S, i-1) + SizeOfMember(S, i-1))
            let next_offset = round_up(align_of_next, curr_offset + curr.byte_size());
            curr_offset = next_offset;
        }
        curr_offset
    }

    pub(crate) fn len(&self) -> usize { self.sized_fields.len() + self.last_unsized.iter().count() }

    pub(crate) fn is_empty(&self) -> bool { self.sized_fields.is_empty() && self.last_unsized.is_none() }

    pub(crate) fn fields(&self) -> impl Iterator<Item = &dyn Field> + Clone {
        let fields = self.sized_fields.iter().map(|f| f as &dyn Field);
        let last = self.last_unsized.iter().map(|f| f as &dyn Field);
        fields.chain(last)
    }

    pub(crate) fn get_field_by_name(&self, canonical_name: &CanonName) -> Option<&dyn Field> {
        self.fields().find(|f| f.name() == canonical_name)
    }

    pub(crate) fn name(&self) -> &CanonName { &self.name }

    pub(crate) fn is_creation_fixed_footprint(&self) -> bool {
        self.sized_fields
            .iter()
            .all(|field| StoreType::is_creation_fixed_footprint(&StoreType::from(field.ty.clone()))) &&
            self.last_unsized.is_none()
    }

    pub(crate) fn contains_atomics(&self) -> bool {
        self.sized_fields.iter().any(|f| f.ty.contains_atomics()) &&
            self.last_unsized.iter().any(|f| f.element_ty.contains_atomics())
    }

    pub(crate) fn is_host_shareable(&self) -> bool {
        self.sized_fields.iter().all(|f| f.ty.is_host_shareable()) &&
            self.last_unsized.iter().all(|f| f.element_ty.is_host_shareable())
    }

    pub(crate) fn sized_fields(&self) -> &[SizedField] { &self.sized_fields }

    pub(crate) fn last_unsized_field(&self) -> &Option<RuntimeSizedArrayField> { &self.last_unsized }
}

/// try register `struct_` if we're currently in a pipeline encoding,
/// otherwise the registration will happen later with a less useful `call_info`
fn try_register_struct(call_info: CallInfo, struct_: &Rc<Struct>) {
    Context::try_with(call_info, |ctx| {
        ctx.struct_registry_mut().register_mentioned_structs_recursively(
            struct_,
            &mut ctx.pool_mut(),
            ctx.latest_user_caller(),
        );
    });
}

impl Display for Struct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {{", &self.name)?;
        for field in self.fields() {
            write!(f, "  ")?;
            if let Some(min_align) = field.custom_min_align() {
                write!(f, "#[min_align({})] ", u32::from(min_align))?
            }
            if let Some(min_size) = field.custom_min_size() {
                write!(f, "#[min_size({min_size})] ")?
            }
            write!(f, "{}: ", &field.name())?;
            write!(f, "{}", &field.ty())?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizedStruct(Rc<Struct>);

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferBlock(Rc<Struct>);

impl TryFrom<BufferBlock> for SizedStruct {
    type Error = StructureDefinitionError;

    fn try_from(block: BufferBlock) -> Result<Self, StructureDefinitionError> {
        let inner = Struct::clone(&block); // a bit inefficient, but i don't want code duplication of the failure cases in here
        SizedStruct::try_from(Struct::new(
            StructKind::Sized,
            inner.name,
            inner.sized_fields,
            inner.last_unsized,
        )?)
    }
}

#[rustfmt::skip]
impl std::ops::Deref for SizedStruct {
    type Target = Rc<Struct>;
    fn deref(&self) -> &Self::Target {&self.0}
}

#[rustfmt::skip]
impl std::ops::Deref for BufferBlock {
    type Target = Rc<Struct>;
    fn deref(&self) -> &Self::Target {&self.0}
}

impl TryFrom<Rc<Struct>> for SizedStruct {
    type Error = StructureDefinitionError;
    fn try_from(t: Rc<Struct>) -> Result<Self, StructureDefinitionError> {
        match t.kind {
            StructKind::Sized => Ok(Self(t)),
            kind => Err(StructureDefinitionError::WrongStructKindForStructType(
                kind,
                std::stringify!(SizedStruct),
            )),
        }
    }
}

pub struct WrongStructKindForStructType(StructKind, &'static str);

impl TryFrom<Rc<Struct>> for BufferBlock {
    type Error = WrongStructKindForStructType;
    fn try_from(t: Rc<Struct>) -> Result<Self, WrongStructKindForStructType> {
        match t.kind {
            StructKind::BufferBlock => Ok(Self(t)),
            kind => Err(WrongStructKindForStructType(kind, std::stringify!(BufferBlock))),
        }
    }
}

/// an error created if a struct contains two or more fields of the same name
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureFieldNamesMustBeUnique {
    pub first_occurence: usize,
    pub second_occurence: usize,
}

fn check_for_duplicate_field_names(
    sized_fields: &[SizedField],
    last_unsized: Option<&RuntimeSizedArrayField>,
) -> Result<(), StructureFieldNamesMustBeUnique> {
    // Brute force search > HashMap for the amount of fields
    // we'd usually deal with.
    let mut duplicate_fields = None;
    for (i, field1) in sized_fields.iter().enumerate() {
        for (j, field2) in sized_fields.iter().enumerate().skip(i + 1) {
            if field1.name == field2.name {
                duplicate_fields = Some((i, j));
                break;
            }
        }
        if let Some(last_unsized) = last_unsized {
            if field1.name == last_unsized.name {
                duplicate_fields = Some((i, sized_fields.len()));
                break;
            }
        }
    }
    match duplicate_fields {
        Some((first_occurence, second_occurence)) => Err(StructureFieldNamesMustBeUnique {
            first_occurence,
            second_occurence,
        }),
        None => Ok(()),
    }
}

impl SizedStruct {
    #[track_caller]
    pub fn new(name: CanonName, sized_fields_nonempty: Vec<SizedField>) -> Result<Self, StructureDefinitionError> {
        let struct_ = Struct::new(StructKind::Sized, name, sized_fields_nonempty, None)?;
        struct_.try_into()
    }

    #[track_caller]
    pub fn new_nonempty(
        name: CanonName,
        mut sized_fields_first: Vec<SizedField>,
        sized_fields_last: SizedField,
    ) -> Result<Self, StructureFieldNamesMustBeUnique> {
        Ok(Self(Struct::new_sized(
            StructKind::Sized,
            name,
            sized_fields_first,
            sized_fields_last,
        )?))
    }

    pub fn byte_size(&self) -> u64 {
        // the `min_byte_size` is the `byte_size` in sized structs
        self.min_byte_size()
    }

    pub fn fields(&self) -> impl Iterator<Item = &SizedField> { self.sized_fields.iter() }

    pub fn get_field_by_name(&self, canonical_name: &CanonName) -> Option<&SizedField> {
        self.fields().find(|f| &f.name == canonical_name)
    }

    pub fn map_field_types(mut self, mut f: impl FnMut(SizedType) -> SizedType) -> SizedStruct {
        self.0 = Rc::new(Struct::clone(&self.0).map_sized_field_types(f));
        self
    }
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone)]
pub enum BufferBlockDefinitionError {
    #[error("buffer block must have at least one field")]
    MustHaveAtLeastOneField,
    #[error("field names of a buffer block must be unique")]
    FieldNamesMustBeUnique(StructureFieldNamesMustBeUnique),
}

impl BufferBlock {
    pub fn new(
        name: CanonName,
        sized_fields: Vec<SizedField>,
        last_unsized: Option<RuntimeSizedArrayField>,
    ) -> Result<Self, BufferBlockDefinitionError> {
        use StructureDefinitionError as E;
        match Struct::new(StructKind::BufferBlock, name, sized_fields, last_unsized) {
            Ok(s) => Ok(BufferBlock(s)),
            Err(e) => Err(match e {
                E::RuntimeSizedArrayAtNonLastField => unreachable!("prevented by `new` signature"),
                E::WrongStructKindForStructType(_, _) => unreachable!("error never created by Struct::new"),
                E::RuntimeSizedArrayNotAllowedInSizedStruct => unreachable!("not a sized struct"),
                E::MustHaveAtLeastOneField(_) => BufferBlockDefinitionError::MustHaveAtLeastOneField,
                E::FieldNamesMustBeUnique(e) => BufferBlockDefinitionError::FieldNamesMustBeUnique(e),
            }),
        }
    }

    pub fn is_constructible(&self) -> bool {
        // even though the wgsl type may be constructible, the glsl "type" is not.
        false
    }

    pub fn is_fixed_footprint(&self) -> bool { self.last_unsized.is_none() }

    pub fn map_sized_field_types(mut self, mut f: impl FnMut(SizedType) -> SizedType) -> BufferBlock {
        self.0 = Rc::new(Struct::clone(&self.0).map_sized_field_types(f));
        self
    }
}

#[doc(hidden)] // internal
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StructKind {
    Sized,
    BufferBlock,
}

impl Display for StructKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StructKind::Sized => "struct",
            StructKind::BufferBlock => "buffer block",
        })
    }
}

/// the precise definition of a struct type wrt. actual `Ident`s rather than just
/// canonical names of fields etc.
pub struct StructDef {
    call_info: CallInfo,
    kind: StructKind,
    name: CanonName,
    ident: Key<Ident>,
    sized_fields: Vec<(Key<Ident>, SizedField)>,
    last_unsized: Option<(Key<Ident>, RuntimeSizedArrayField)>,
}

impl StructDef {
    pub fn new_for_struct(s: &Rc<Struct>, idents: &mut PoolRefMut<Ident>, call_info: CallInfo) -> Self {
        StructDef {
            call_info,
            kind: s.kind,
            name: s.name.clone(),
            ident: Ident::auto_in_pool(s.name.to_string(), idents),
            sized_fields: s
                .sized_fields
                .iter()
                .map(|f| (Ident::auto_in_pool(f.name.to_string(), idents), f.clone()))
                .collect(),
            last_unsized: s
                .last_unsized
                .as_ref()
                .map(|f| (Ident::auto_in_pool(f.name.to_string(), idents), f.clone())),
        }
    }

    pub fn call_info(&self) -> CallInfo { self.call_info }

    pub fn canonical_name(&self) -> &CanonName { &self.name }

    pub fn ident(&self) -> Key<Ident> { self.ident }

    pub fn fields(&self) -> impl Iterator<Item = (&Key<Ident>, &dyn Field)> {
        let fields = self.sized_fields.iter().map(|(i, f)| (i, f as &dyn Field));
        let last = self.last_unsized.iter().map(|(i, f)| (i, f as &dyn Field));
        fields.chain(last)
    }

    pub fn get_field_by_name(&self, canonical_name: &CanonName) -> Option<(&Key<Ident>, &dyn Field)> {
        self.fields().find(|(ident, f)| f.name() == canonical_name)
    }
}

#[derive(Default)]
pub struct StructRegistry {
    /// "topologically sorted" list of structure definitions
    ///
    /// if structure `b`'s fields reference a structure `a` in any way, `a` appears
    /// before `b` in this list.
    defs: Vec<(Rc<Struct>, StructDef)>,
}

impl StructRegistry {
    pub fn get(&self, s: &Rc<Struct>) -> Option<&StructDef> {
        //TODO(release) this is quite inefficient because of the struct equals check on _every_ registered struct
        // consider using a different datastructure + representing the topological sort differently
        self.defs.iter().find_map(|(o, def)| (o == s).then_some(def))
    }

    /// returns an iterator that goes through the structs in a "topologically sorted" way.
    /// this means if structure `b`'s fields reference a structure `a` in any way, `a` appears
    /// before `b` in this list.
    pub fn iter_topo_sorted(&self) -> impl Iterator<Item = &StructDef> { self.defs.iter().map(|(_, def)| def) }


    fn validate_custom_align_and_size(s: &Rc<Struct>) -> Result<(), LayoutError> {
        for field in s.sized_fields() {
            if let Some(c_align) = field.custom_min_align.map(u64::from) {
                let ty_align = field.ty.align();
                if c_align < ty_align {
                    return Err(LayoutError::CustomAlignmentTooSmall {
                        custom: c_align,
                        required: ty_align,
                        ty: field.ty.clone().into(),
                    });
                }
            }

            if let Some(c_size) = field.custom_min_size {
                let ty_size = field.ty.byte_size();
                if c_size < ty_size {
                    return Err(LayoutError::CustomSizeTooSmall {
                        custom: c_size,
                        required: ty_size,
                        ty: field.ty.clone().into(),
                    });
                }
            }
        }
        if let Some(field) = &s.last_unsized_field() {
            if let Some(c_align) = field.custom_min_align.map(u64::from) {
                let ty_align = field.element_ty().align();
                if c_align < ty_align {
                    return Err(LayoutError::CustomAlignmentTooSmall {
                        custom: c_align,
                        required: ty_align,
                        ty: field.ty().clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// only registers this one struct (if it wasn't registered before),
    /// not any structures that are used within that struct's fields.
    ///
    /// this function must not be public, because that can be used to violate
    /// the topological sort.
    /*not pub*/
    fn register_single_struct(&mut self, s: &Rc<Struct>, idents: &mut PoolRefMut<Ident>, call_info: CallInfo) -> bool {
        if !self.contains(s) {
            Context::try_with(call_info, |ctx| {
                if let Err(e) = Self::validate_custom_align_and_size(s) {
                    ctx.push_error(e.into())
                }
            });
            self.defs
                .push((s.clone(), StructDef::new_for_struct(s, idents, call_info)));
            true
        } else {
            false
        }
    }

    /// registers `s` as well as any other structs mentioned in the fields of
    /// this struct (recursively) if they aren't already registered.
    pub fn register_mentioned_structs_recursively(
        &mut self,
        s: &Rc<Struct>,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        // Note: ORDER IMPORTANT
        // first iterate, then insert (this ensures topological sortedness)
        for field in s.fields() {
            self.find_and_register_new_structs_used_in_store_type(&field.ty(), idents, call_info);
        }
        self.register_single_struct(s, idents, call_info);
    }

    pub fn find_and_register_new_structs_used_in_type(
        &mut self,
        t: &Type,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        match t {
            Type::Unit => (),
            Type::Ptr(_, s, _) | Type::Ref(_, s, _) | Type::Store(s) => {
                self.find_and_register_new_structs_used_in_store_type(s, idents, call_info)
            }
        }
    }

    pub fn find_and_register_new_structs_used_in_store_type(
        &mut self,
        t: &StoreType,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        match t {
            StoreType::Handle(_) => (),
            StoreType::RuntimeSizedArray(s) | StoreType::Sized(s) => {
                self.find_and_register_new_structs_used_in_sized_type(s, idents, call_info)
            }
            StoreType::BufferBlock(s) => self.register_mentioned_structs_recursively(s, idents, call_info),
        }
    }

    pub fn find_and_register_new_structs_used_in_sized_type(
        &mut self,
        t: &SizedType,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        match t {
            SizedType::Vector(_, _) | SizedType::Matrix(_, _, _) | SizedType::Atomic(_) => (),
            SizedType::Array(e, _) => self.find_and_register_new_structs_used_in_sized_type(e, idents, call_info),
            SizedType::Structure(e) => self.register_mentioned_structs_recursively(e, idents, call_info),
        }
    }

    pub fn contains(&mut self, s: &Rc<Struct>) -> bool { self.defs.iter().any(|(x, _)| x == s) }

    pub fn definitions(&self) -> &[(Rc<Struct>, StructDef)] { &self.defs }
}
