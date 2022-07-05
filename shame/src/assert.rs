
use shame_graph::Any;

/// asserts a condition and pushes the error to the shader recording errors 
/// of the current recording context
#[macro_export]
macro_rules! rec_assert {
    ($cond: expr $(,)?) => {
        crate::assert::assert_string($cond, format!("({})", stringify!($cond)))
    };
    ($cond: expr, $s: expr) => {
        crate::assert::assert_string($cond, $s)
    };
}

#[track_caller]
pub fn rec_error(err: shame_graph::Error) {
    shame_graph::Context::with(|ctx| {
        ctx.push_error(err)
    });
}

#[track_caller]
pub fn assert_string(cond: bool, s: impl AsRef<str>) -> Option<shame_graph::Any> {
    match cond {
        true => None,
        false => {
            rec_error(shame_graph::Error::AssertionFailed(s.as_ref().to_string()));
            Some(Any::not_available())
        }
    }
}