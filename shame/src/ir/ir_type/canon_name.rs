use std::{borrow::Cow, fmt::Display, sync::Arc};

use crate::common::ignore_eq::IgnoreInEqOrdHash;


/// canonical name of something, as opposed to `Ident` which would be an
/// identifier that changes between creation and final appearance in a shader.
///
/// The canonical name of a struct field is used to refer to a
/// given field unambiguously within the `shame` graph.
///
/// In most cases the canonical name is the obvious name for something, for
/// example in a struct
/// ```
/// struct Foo {
///     bar: float,
///     vec3: float,
/// }
/// ```
/// `bar` and `vec3` are the canonical names of these two fields. However,
/// when translating this into shader code, `vec3` cannot be kept as an
/// identifier because it is a predefined type in some target languages.
/// It must be modified to something like `s_vec3`.
///
/// to summarize:
/// - `vec3` = canonical name (which cannot be used in the shader in this case)
/// - `vec3` = identifier (before disambiguation)
/// - `s_vec3` = final identifier
///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonName(pub Cow<'static, str>);

impl std::ops::Deref for CanonName {
    type Target = str;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl From<&'static str> for CanonName {
    fn from(x: &'static str) -> Self { CanonName(x.into()) }
}

impl From<&'static str> for IgnoreInEqOrdHash<CanonName> {
    fn from(t: &'static str) -> Self { IgnoreInEqOrdHash(t.into()) }
}

impl From<String> for CanonName {
    fn from(x: String) -> Self { CanonName(x.into()) }
}

impl From<String> for IgnoreInEqOrdHash<CanonName> {
    fn from(t: String) -> Self { IgnoreInEqOrdHash(t.into()) }
}

impl Display for CanonName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
