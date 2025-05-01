use shame as sm;
use crate::conversion::ShameToWgpuError;

/// this trait connects the shame binding types `sm::Binding` to their
/// `wgpu::BindingResource`, `wgpu::BindingType` types
pub trait AsBindingResource: sm::Binding {
    type BindingResource<'a>: ?Sized;

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a>;

    fn bind_group_layout_entry(
        device: &wgpu::Device,
        binding_index: u32,
    ) -> Result<wgpu::BindGroupLayoutEntry, ShameToWgpuError> {
        let vertex_writable_storage_enabled = device.features().contains(wgpu::Features::VERTEX_WRITABLE_STORAGE);
        let sm_layout =
            sm::results::BindingLayout::from_ty_with_max_visibility::<Self>(vertex_writable_storage_enabled);
        crate::conversion::binding_layout(binding_index, &sm_layout)
    }
}

impl<T, AS, const DYNAMIC_OFFSET: bool> AsBindingResource for sm::Buffer<T, AS, DYNAMIC_OFFSET>
where
    Self: sm::Binding,
    T: sm::GpuStore + sm::NoHandles + sm::NoAtomics + sm::NoBools,
    AS: sm::BufferAddressSpace,
{
    type BindingResource<'a> = wgpu::BufferBinding<'a>;

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(resource.clone())
    }
}

impl<T, AS, AM, const DYNAMIC_OFFSET: bool> AsBindingResource for sm::BufferRef<T, AS, AM, DYNAMIC_OFFSET>
where
    Self: sm::Binding,
    T: sm::GpuStore + sm::NoHandles + sm::NoBools,
    AS: sm::BufferAddressSpace,
    AM: sm::AccessModeReadable,
{
    type BindingResource<'a> = wgpu::BufferBinding<'a>;

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(resource.clone())
    }
}

impl<M: sm::SamplingMethod> AsBindingResource for sm::Sampler<M> {
    type BindingResource<'a> = wgpu::Sampler;

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Sampler(resource)
    }
}

impl<Format, Coords, SPP> AsBindingResource for sm::Texture<Format, Coords, SPP>
where
    Coords: sm::TextureCoords + sm::SupportsSpp<SPP>,
    Format: sm::SamplingFormat + sm::SupportsSpp<SPP> + sm::SupportsCoords<Coords>,
    SPP: sm::Spp,
{
    type BindingResource<'a> = wgpu::TextureView;

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::TextureView(resource)
    }
}

impl<Format, Coords, const N: u32> AsBindingResource for sm::TextureArray<Format, N, Coords>
where
    Coords: sm::TextureCoords + sm::LayerCoords,
    Format: sm::SamplingFormat + sm::SupportsCoords<Coords>,
{
    type BindingResource<'a> = [&'a wgpu::TextureView];

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::TextureViewArray(resource)
    }
}

impl<Format, Coords, Access> AsBindingResource for sm::StorageTexture<Format, Coords, Access>
where
    Coords: sm::StorageTextureCoords,
    Format: sm::StorageTextureFormat<Access> + sm::SupportsCoords<Coords>,
    Access: sm::AccessMode,
{
    type BindingResource<'a> = wgpu::TextureView;

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::TextureView(resource)
    }
}

impl<Format, Coords, Access, const N: u32> AsBindingResource for sm::StorageTextureArray<Format, N, Coords, Access>
where
    Coords: sm::StorageTextureCoords + sm::LayerCoords,
    Format: sm::StorageTextureFormat<Access> + sm::SupportsCoords<Coords>,
    Access: sm::AccessMode,
{
    type BindingResource<'a> = [&'a wgpu::TextureView];

    fn binding_resource<'a>(resource: &'a Self::BindingResource<'a>) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::TextureViewArray(resource)
    }
}
