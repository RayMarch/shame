use std::marker::PhantomData;

use super::conversion::ShameToWgpuError;


pub struct BindGroupLayout<T: AsBindGroupLayout> {
    layout: wgpu::BindGroupLayout,
    phantom: PhantomData<T>,
}

impl<T: AsBindGroupLayout> BindGroupLayout<T> {
    pub fn from_parts_unchecked(layout: wgpu::BindGroupLayout) -> Self {
        Self {
            layout,
            phantom: PhantomData,
        }
    }

    pub fn new_bind_group(&self, device: &wgpu::Device, res: T::Resources<'_>) -> wgpu::BindGroup {
        T::create_bind_group(&self.layout, device, res)
    }
}

pub trait AsBindGroupLayout: for<'it> From<shame::BindingIter<'it>> {
    type Resources<'a>;
    fn create_bind_group_layout(device: &wgpu::Device) -> Result<BindGroupLayout<Self>, ShameToWgpuError>;
    fn create_bind_group(
        layout: &wgpu::BindGroupLayout,
        gpu: &wgpu::Device,
        res: Self::Resources<'_>,
    ) -> wgpu::BindGroup;
}

#[macro_export]
macro_rules! bind_group {
    (
        $vis: vis struct $Struct: ident {
            $($field_vis: vis $field: ident: $field_ty: ty),* $(,)?
        }
    ) => {
        $vis struct $Struct {
            $($field_vis $field: $field_ty),*
        }

        impl From<::shame::BindingIter<'_>> for $Struct {
            fn from(mut it: ::shame::BindingIter) -> Self {
                $Struct {
                    $($field: it.next()),*
                }
            }
        }

        $crate::__reexport::concat_idents! {TResources = $Struct, Resources {
            $vis struct TResources<'a> {
                $(pub $field: <$field_ty as $crate::binding::AsBindingResource>::BindingResource<'a>),*
            }

            impl $crate::bind_group::AsBindGroupLayout for $Struct {
                type Resources<'a> = TResources<'a>;

                fn create_bind_group_layout(device: &wgpu::Device) -> Result<$crate::bind_group::BindGroupLayout<Self>, $crate::conversion::ShameToWgpuError> {
                    use $crate::binding::AsBindingResource as _;
                    use $crate::bind_group::BindGroupLayout;

                    let vertex_writable_storage_enabled = device.features().contains(wgpu::Features::VERTEX_WRITABLE_STORAGE);
                    let mut i = 0..;
                    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            $(<$field_ty>::bind_group_layout_entry(device, i.next().unwrap_or_default())?),*
                        ],
                    });
                    Ok(BindGroupLayout::from_parts_unchecked(bgl))
                }

                fn create_bind_group(layout: &wgpu::BindGroupLayout, gpu: &wgpu::Device, res: TResources) -> wgpu::BindGroup {
                    use $crate::binding::AsBindingResource as _;

                    let mut i = 0..;
                    gpu.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout,
                        entries: &[$(
                            wgpu::BindGroupEntry {
                                binding: i.next().unwrap_or(u32::MAX),
                                resource: <$field_ty>::binding_resource(&res. $field),
                            }
                        ),*],
                    })
                } // fn
            } // impl
        }}
    };
}
