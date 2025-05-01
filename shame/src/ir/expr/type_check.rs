use crate::{
    common::small_vec_actual::SmallVec,
    ir::ir_type::{Len, Len2, ScalarType, SizedType, StoreType},
    ir::{self, Type},
};
use std::fmt::Write;
use std::{borrow::Cow, fmt::Display};

pub trait TypeCheck {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature>;
}

/// define type signatures for recording-time type checking
///
/// the specified signatures create a match statement that decides whether the
/// call argument types are valid or not.
/// The signatures are also stringified and put into the error message if
/// invalid arguments were provided.
///
/// optionally the `$name`, `$comment_below` and `$fmt` metavariables can be used
/// to modify how the error is displayed.
///
/// - `$name` allows a `pat` to be used as the name, this is useful for
///    expressions like `Bitcast(T)` where the name "T" shows up in the signature
///    match arms, but is not defined anywhere. use `name: Bitcast(T)` in that
///    case to present the definition of "T" to the user.
/// - `$comment_below` to add a string below the match pattern signatures, for
///     example to provide context as to how it is to be interpreted
/// - `$fmt` to apply one of the formatting options to make the match arms
///     less confusing when seen out of context. For example, remove asterisks
///     "*" used for dereferencing
/// - `$force_in_type` forces the matched argument array to only match arguments of
///     that type. This is useful for example when matching empty [] brackets
///     since the rust compiler wouldn't know what type the input shorthand should have
#[doc(hidden)] // internal api
#[macro_export]
macro_rules! sig {
    (
        $({
            $(name: $name: pat,)?
            $(comment_below: $comment_below: expr,)?
            $(fmt: $fmt: expr,)?
        },)?
        $(
            [] $(if $cond: expr)? => $result: expr
        ),* $(,)?
    ) => {
        // this just matches the empty [] pattern, see the real macro below
        sig!({
            $($(name: $name,)?)?
            $($(comment_below: $comment_below,)?)?
            $($(fmt: $fmt,)?)?
            force_in_type: $crate::ir::ir_type::Type,
        },
            $([] $(if $cond)? => $result),*
        )
    };

    (
        $({
            $(name: $name: pat,)?
            $(comment_below: $comment_below: expr,)?
            $(fmt: $fmt: expr,)?
            $(force_in_type: $force_in_type: ty,)?
        },)?
        $(
            $pattern: pat $(if $cond: expr)? => $result: expr
        ),* $(,)?
    ) => {move |self_: &dyn std::fmt::Debug, args: &[$crate::ir::ir_type::Type]| {
        use $crate::common::small_vec::SmallVec;
        use $crate::ir::expr::type_check::TypeShorthand;
        use $crate::ir::expr::type_check::SignatureStrings;
        use $crate::ir::expr::type_check::shorthand_level_for_args;

        // map types to type shorthands for simpler syntax on the macro-callsite,
        // then create `SmallVec` from that map-iterator to obtain a matchable
        // slice from `SmallVec`'s `std::ops::Deref`.
        let vec
        $($(: Option<SmallVec<&$force_in_type, 4>>)?)?
         = SmallVec::<_, 4>::from_opt_iter(args.iter().map(TypeShorthand::shorthand_for));

        let shorthand_level = shorthand_level_for_args(&vec);

        let error = || {
            const SIG_STRINGS: &[&str] = &[
                $(std::stringify!($pattern $(if $cond )?=> $result)),*
            ];

            let expression_name = match () {
                $($(() => stringify!($name).into(),)?)?
                _ => format!("{:?}", self_).into(),
            };

            let comment = match () {
                $($(() => Some(String::from($comment_below)),)?)?
                _ => None,
            };

            let signature_formatting = match () {
                $($(() => Some($fmt),)?)?
                _ => None,
            };

            if false {
                match unreachable!("enable syntax highlighting") {
                    $($($name => {})?)?
                    _ => {}
                }
            }

            Err($crate::ir::expr::type_check::NoMatchingSignature {
                expression_name,
                arguments: args.iter().cloned().collect(),
                shorthand_level,
                allowed_signatures: SignatureStrings::Static(SIG_STRINGS),
                signature_formatting,
                comment,
            })
        };

        let Some(vec) = vec else {
            return error()
        };

        // match the caller-defined patterns, if none matches return an error
        // containing the match arms as strings.
        let ty = TypeShorthand::to_type(match *vec {
            $($pattern $(if $cond)? => ($result).clone(),)*
            _ => return error()
        });
        Ok(ty)
    }};
}

#[derive(Debug, Clone)]
pub enum SignatureStrings {
    Static(&'static [&'static str]),
    Dynamic(Vec<String>),
}

impl SignatureStrings {
    pub fn is_empty(&self) -> bool {
        match self {
            SignatureStrings::Static(s) => s.is_empty(),
            SignatureStrings::Dynamic(s) => s.is_empty(),
        }
    }

    pub fn collect(&self) -> Vec<String> {
        match self {
            SignatureStrings::Static(s) => s.iter().map(|s| s.to_string()).collect(),
            SignatureStrings::Dynamic(s) => s.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NoMatchingSignature {
    pub expression_name: Cow<'static, str>,
    pub arguments: SmallVec<Type, 4>,
    pub allowed_signatures: SignatureStrings,
    pub shorthand_level: TypeShorthandLevel,
    pub signature_formatting: Option<SigFormatting>,
    pub comment: Option<String>,
}

impl NoMatchingSignature {
    pub fn empty_with_name(expr_name: Cow<'static, str>, args: &[ir::Type]) -> Self {
        NoMatchingSignature {
            expression_name: expr_name,
            arguments: args.iter().cloned().collect(),
            allowed_signatures: SignatureStrings::Static(&[]),
            shorthand_level: Default::default(),
            signature_formatting: None,
            comment: None,
        }
    }

    pub fn empty_with_name_and_comment(expr_name: Cow<'static, str>, comment: String, args: &[ir::Type]) -> Self {
        NoMatchingSignature {
            expression_name: expr_name,
            arguments: args.iter().cloned().collect(),
            allowed_signatures: SignatureStrings::Static(&[]),
            shorthand_level: Default::default(),
            signature_formatting: None,
            comment: Some(comment),
        }
    }

    pub fn concat(mut self, o: NoMatchingSignature) -> NoMatchingSignature {
        self.expression_name = format!("{}, {}", self.expression_name, o.expression_name).into();
        let mut sigs = match self.allowed_signatures {
            SignatureStrings::Static(strs) => strs.iter().map(ToString::to_string).collect(),
            SignatureStrings::Dynamic(vec) => vec,
        };
        match o.allowed_signatures {
            SignatureStrings::Static(strs) => sigs.extend(strs.iter().map(ToString::to_string)),
            SignatureStrings::Dynamic(vec) => sigs.extend(vec),
        }
        self.allowed_signatures = SignatureStrings::Dynamic(sigs);
        self.shorthand_level = TypeShorthandLevel::most_verbose(self.shorthand_level, o.shorthand_level);
        self.signature_formatting = None;
        self.comment = match (self.comment, o.comment) {
            (None, c) | (c, None) => c,
            (Some(c0), Some(c1)) => Some(c0 + "\n" + &*c1),
        };
        self
    }
}

impl std::error::Error for NoMatchingSignature {}

impl Display for NoMatchingSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use {SigFormatting::*, TypeShorthandLevel as Lv};
        let name = &self.expression_name;
        write!(f, "no matching function call for `{name}` ")?;
        write!(f, "with argument types\n\n[")?;
        let mut first = true;
        for arg in self.arguments.iter() {
            if !first {
                write!(f, ", ")?;
            }
            first = false;

            // if a certain "shorthand" was applied
            // (in this context shorthand means "Type(None, Vector(X2, F32))" is displayed as "Vector(X2, F32)" via the "TypeShorthandLevel::StoreType" shortening)
            // try to apply that same shortening to the args if possible.
            // If not, display a more detailed type, or the entire type.
            match (&self.shorthand_level, arg) {
                (Lv::Type, _) => write!(f, "{arg:?}")?,
                (Lv::StoreType, Type::Store(store_type)) => write!(f, "{store_type:?}")?,
                (Lv::SizedType, Type::Store(StoreType::Sized(sized_type))) => write!(f, "{sized_type:?}")?,
                (Lv::ScalarType, Type::Store(StoreType::Sized(SizedType::Vector(Len::X1, scalar_type)))) => {
                    write!(f, "{scalar_type:?}")?
                }
                (Lv::ScalarType, Type::Store(store_type)) => write!(f, "{store_type:?}")?,
                _ => write!(f, "{arg:?}")?,
            }
        }
        writeln!(f, "]\n")?;
        if !self.allowed_signatures.is_empty() {
            writeln!(
                f,
                "Accepted type signatures of `{name}` are:\n(in match-pattern syntax)"
            )?;
            writeln!(f)?;
            for sig in self.allowed_signatures.collect() {
                let sig_str = match &self.signature_formatting {
                    Some(fmt) => match fmt {
                        RemoveAsterisksAndClone => sig.to_string().replace('*', "").replace(".clone()", ""),
                    },
                    None => sig.to_string(),
                };
                writeln!(f, "{},", sig_str.replace('\n', " "))?;
            }
            writeln!(f)?;
        }
        if let Some(comment) = &self.comment {
            writeln!(f, "{comment}\n")?;
        }
        writeln!(
            f,
            "If you called `{name}` with `shame::Any` instances this means the dynamic type of \
            these instances does not match what `{name}` expects.\nYou can access the dynamic \
            type of a `shame::Any` via `.ty()` for conditional recording/debugging.\n\n\
            If instead you used the regular high-level `shame` functions and types, then this is likely a bug in `shame`. \
            In that case please file an issue on github using this link:\n\
            https://github.com/raymarch/shame/issues/new?title=Dynamic%20type%20check%20Issue&body=Hi%2C%20I%20got%20the%20following%20error"
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SigFormatting {
    /// replace all "*" with ""
    RemoveAsterisksAndClone,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum TypeShorthandLevel {
    #[default]
    Type,
    StoreType,
    SizedType,
    ScalarType,
}

impl TypeShorthandLevel {
    /// the more verbose of the two
    /// `Level::Type` is the most verbose
    /// `Level::Scalar` is the least verbose
    ///
    /// in the future some shorthands may have same verbostiy
    fn most_verbose(a: Self, b: Self) -> Self {
        use TypeShorthandLevel::*;
        let verbosity = |x: TypeShorthandLevel| match x {
            Type => 3,
            StoreType => 2,
            SizedType => 1,
            ScalarType => 0,
        };
        match verbosity(a) > verbosity(b) {
            true => a,
            false => b,
        }
    }
}

pub(crate) const fn shorthand_level_for_args<T: TypeShorthand, const N: usize>(
    args: &Option<SmallVec<&T, N>>,
) -> TypeShorthandLevel {
    T::SHORT_LEVEL
}

pub(crate) trait TypeShorthand: Sized {
    const SHORT_LEVEL: TypeShorthandLevel;
    fn to_type(self) -> Type;
    fn shorthand_for(t: &Type) -> Option<&Self>;
}

impl TypeShorthand for Type {
    const SHORT_LEVEL: TypeShorthandLevel = TypeShorthandLevel::Type;
    fn to_type(self) -> Type { self }
    fn shorthand_for(t: &Type) -> Option<&Self> { Some(t) }
}

impl TypeShorthand for StoreType {
    const SHORT_LEVEL: TypeShorthandLevel = TypeShorthandLevel::StoreType;
    fn to_type(self) -> Type { Type::Store(self) }
    fn shorthand_for(t: &Type) -> Option<&Self> {
        match t {
            Type::Store(store_type) => Some(store_type),
            _ => None,
        }
    }
}

impl TypeShorthand for SizedType {
    const SHORT_LEVEL: TypeShorthandLevel = TypeShorthandLevel::SizedType;
    fn to_type(self) -> Type { Type::Store(StoreType::Sized(self)) }
    fn shorthand_for(t: &Type) -> Option<&Self> {
        match t {
            Type::Store(StoreType::Sized(sized_type)) => Some(sized_type),
            _ => None,
        }
    }
}

impl TypeShorthand for ScalarType {
    const SHORT_LEVEL: TypeShorthandLevel = TypeShorthandLevel::ScalarType;
    fn to_type(self) -> Type { Type::Store(StoreType::Sized(SizedType::Vector(Len::X1, self))) }
    fn shorthand_for(t: &Type) -> Option<&Self> {
        match t {
            Type::Store(StoreType::Sized(SizedType::Vector(Len::X1, scalar))) => Some(scalar),
            _ => None,
        }
    }
}

#[doc(hidden)] // internal api
#[macro_export]
macro_rules! same {
    ($($head: ident $($tail: ident)+);*) => {
        true $($(&& $head == $tail)+)*
    };
}
