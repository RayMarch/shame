use std::fmt::Display;

use crate::ir::ir_type::Type;

// https://www.w3.org/TR/WGSL/#address-space
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(missing_docs)] // runtime api
pub enum AddressSpace {
    #[default]
    Function,
    Thread, //aka Private
    WorkGroup,
    Uniform,
    Storage,
    Handle,
    PushConstant,
    Output,
}

impl Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AddressSpace::Function => "function",
            AddressSpace::Thread => "thread",
            AddressSpace::WorkGroup => "workgroup",
            AddressSpace::Uniform => "uniform",
            AddressSpace::Storage => "storage",
            AddressSpace::Handle => "handle",
            AddressSpace::PushConstant => "push_constant",
            AddressSpace::Output => "output",
        })
    }
}

impl AddressSpace {
    /// a suffix that is sometimes added to identifiers when trying to hint at a certain address space
    ///
    /// for example, in the WGSL backend the `atomicCompareExchangeWeak` specializations add the chosen
    /// `atomic_ptr` argument's address space as a suffix to the function name
    pub(crate) fn ident_suffix(&self) -> &'static str {
        match self {
            AddressSpace::Function => "fn",
            AddressSpace::Thread => "thr",
            AddressSpace::WorkGroup => "workg",
            AddressSpace::Uniform => "uni",
            AddressSpace::Storage => "sto",
            AddressSpace::Handle => "han",
            AddressSpace::PushConstant => "pc",
            AddressSpace::Output => "out",
        }
    }
}

impl AddressSpace {
    #[allow(missing_docs)] // runtime api
    pub fn is_writeable(&self) -> bool {
        self.supports_access(AccessMode::ReadWrite) || self.supports_access(AccessMode::Write)
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_readable(&self) -> bool {
        self.supports_access(AccessMode::ReadWrite) || self.supports_access(AccessMode::Read)
    }

    #[allow(missing_docs)] // runtime api
    #[rustfmt::skip]
    pub fn default_access_mode(&self) -> AccessMode {
        use AddressSpace as AS;
        use AccessMode as AM;
        match self {
            AS::Function      => AM::ReadWrite,
            AS::Thread       => AM::ReadWrite,
            AS::WorkGroup     => AM::ReadWrite,
            AS::Uniform       => AM::Read,
            AS::Storage       => AM::Read,
            AS::Handle        => AM::Read,
            AS::PushConstant  => AM::Read,
            AS::Output        => AM::Write,
        }
    }

    #[allow(missing_docs)] // runtime api
    #[rustfmt::skip]
    pub fn supports_access(&self, am: AccessMode) -> bool {
        use AddressSpace as AS;
        use AccessMode as AM;
        match (am, self) {
            (           AM::ReadWrite, AS::Function    ) |
            (           AM::ReadWrite, AS::Thread     ) |
            (           AM::ReadWrite, AS::WorkGroup   ) |
            (AM::Read                , AS::Uniform     ) |
            (AM::Read | AM::ReadWrite, AS::Storage     ) |
            (AM::Read                , AS::Handle      ) |
            (AM::Read                , AS::PushConstant) |
            (AM::Write               , AS::Output) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Indirection {
    Ptr,
    Ref,
}

#[allow(missing_docs)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// subset of `AccessMode` that is readable
///
/// implements `Into<AccessMode>`
#[allow(missing_docs)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessModeReadable {
    Read,
    ReadWrite,
}

impl From<AccessModeReadable> for AccessMode {
    fn from(value: AccessModeReadable) -> Self {
        match value {
            AccessModeReadable::Read => AccessMode::Read,
            AccessModeReadable::ReadWrite => AccessMode::ReadWrite,
        }
    }
}

impl AccessMode {
    /// `true` if read or readwrite
    pub fn is_readable(self) -> bool {
        match self {
            AccessMode::Read => true,
            AccessMode::Write => false,
            AccessMode::ReadWrite => true,
        }
    }

    /// `true` if write or readwrite
    pub fn is_writeable(self) -> bool {
        match self {
            AccessMode::Read => false,
            AccessMode::Write => true,
            AccessMode::ReadWrite => true,
        }
    }
}

impl Display for AccessMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AccessMode::Read => "read",
            AccessMode::Write => "write",
            AccessMode::ReadWrite => "read_write",
        })
    }
}
