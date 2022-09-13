/// define the mirror module at macro call site to circumvent the orphan rule
/// for the declared traits
#[cfg(feature = "mirror")]
#[macro_export]
macro_rules! define_mirror_module {
    ($module_name: ident) => {
        mod $module_name {
            $crate::define_mirror_traits!{}
        }
    };
    () => {
        $crate::define_mirror_traits!{}
    };
}

/// define the mirror traits at macro call site to circumvent the orphan rule
/// for the declared traits
#[cfg(feature = "mirror")]
#[macro_export]
macro_rules! define_mirror_traits {
    () => {

    pub trait Host {
        type Device: shame::Fields;
        fn as_bytes(&self) -> &[u8];
    }

    pub trait Device {
        type Host: Host + ?Sized;
    }

};
}
