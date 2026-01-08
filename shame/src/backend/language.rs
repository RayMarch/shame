/// the language that the generated shaders should use
///
/// The choice of target language may restrict certain features.
/// These restrictions will be communicated via [`EncodingErrors`].
///
/// [`EncodingErrors`]: crate::EncodingErrors
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Language {
    /// the WebGPU Shader Language
    ///
    /// this language can be transpiled to `spir-v` via `https://crates.io/crates/naga`
    ///
    /// see https://www.w3.org/TR/WGSL/
    Wgsl,
    // SpirV
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Wgsl => write!(f, "wgsl"),
        }
    }
}
