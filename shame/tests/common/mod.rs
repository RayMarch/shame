//shame/tests/shared.rs

/// dirty way to simplify shader code according to some made up rules, just so writing the tests is more convenient.
/// in the future, using a tokenizer might be more appropriate
#[allow(unused)]
pub fn simplify_code(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    s.trim().chars().fold('\n', |prev, c| {
        match (prev, c) {
            ('\n', '\n') | ('\n', ' ') | (' ', ' ') => (), //skip
            _ => out.push(c),
        }
        c
    });
    out
}

#[allow(unused)]
#[macro_export]
macro_rules! assert_eq_code {
    ($a: expr, $b: expr) => {
        assert_eq!(
            $crate::common::simplify_code($a),
            $crate::common::simplify_code($b)
        )
    };
}
