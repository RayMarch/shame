pub trait TextureViewExt: std::borrow::Borrow<wgpu::TextureView> {
    /// `self` as a color attachment
    ///
    /// always returns `Some`, since it needs to be wrapped in `Some` when passing
    /// it to wgpu anyways.
    fn attach_as_color(
        &self,
        load: wgpu::LoadOp<wgpu::Color>,
        store: wgpu::StoreOp,
    ) -> Option<wgpu::RenderPassColorAttachment> {
        Some(wgpu::RenderPassColorAttachment {
            view: self.borrow(),
            resolve_target: None,
            ops: wgpu::Operations { load, store },
        })
    }

    /// `self` as a color attachment with a resolve target
    ///
    /// always returns `Some`, since it needs to be wrapped in `Some` when passing
    /// it to wgpu anyways.
    fn attach_as_color_and_resolve<'a>(
        &'a self,
        load: wgpu::LoadOp<wgpu::Color>,
        store: wgpu::StoreOp,
        resolve_target: &'a wgpu::TextureView,
    ) -> Option<wgpu::RenderPassColorAttachment<'a>> {
        Some(wgpu::RenderPassColorAttachment {
            view: self.borrow(),
            resolve_target: Some(resolve_target),
            ops: wgpu::Operations { load, store },
        })
    }

    /// `self` as a depth attachment
    ///
    /// always returns `Some`, since it needs to be wrapped in `Some` when passing
    /// it to wgpu anyways.
    fn attach_as_depth(
        &self,
        ops: Option<(wgpu::LoadOp<f32>, wgpu::StoreOp)>,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment> {
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: self.borrow(),
            depth_ops: ops.map(|(load, store)| wgpu::Operations { load, store }),
            stencil_ops: None,
        })
    }

    /// `self` as a stencil attachment
    ///
    /// always returns `Some`, since it needs to be wrapped in `Some` when passing
    /// it to wgpu anyways.
    fn attach_as_stencil(
        &self,
        ops: Option<(wgpu::LoadOp<u32>, wgpu::StoreOp)>,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment> {
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: self.borrow(),
            depth_ops: None,
            stencil_ops: ops.map(|(load, store)| wgpu::Operations { load, store }),
        })
    }

    /// `self` as a stencil attachment
    ///
    /// always returns `Some`, since it needs to be wrapped in `Some` when passing
    /// it to wgpu anyways.
    fn attach_as_depth_stencil(
        &self,
        depth_ops: Option<(wgpu::LoadOp<f32>, wgpu::StoreOp)>,
        stencil_ops: Option<(wgpu::LoadOp<u32>, wgpu::StoreOp)>,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment> {
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: self.borrow(),
            depth_ops: depth_ops.map(|(load, store)| wgpu::Operations { load, store }),
            stencil_ops: stencil_ops.map(|(load, store)| wgpu::Operations { load, store }),
        })
    }
}

impl TextureViewExt for wgpu::TextureView {}
