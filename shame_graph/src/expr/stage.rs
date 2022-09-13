#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// FIXME: this is a quick and dirty addition to make error detection in conditional blocks
/// easier before the `Stage` information becomes fully available in `shame_graph`
pub enum Stage {
    Vertex,
    Fragment,
    Uniform,
    NotAvailable,
}

impl std::ops::BitAnd for Stage {
    type Output = Stage;

    fn bitand(self, rhs: Self) -> Self::Output {
        use Stage::*;
        match (self, rhs) {
            (Uniform, x) => x,
            (x, Uniform) => x,

            (NotAvailable, _) => NotAvailable,
            (_, NotAvailable) => NotAvailable,

            (Vertex, Fragment) => NotAvailable,
            (Fragment, Vertex) => NotAvailable,
            (Vertex, Vertex) => Vertex,
            (Fragment, Fragment) => Fragment,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Compute,
}

impl From<ShaderKind> for Stage {
    fn from(k: ShaderKind) -> Self {
        match k {
            ShaderKind::Vertex => Stage::Vertex,
            ShaderKind::Fragment => Stage::Fragment,
            ShaderKind::Compute => Stage::Uniform,
        }
    }
}
