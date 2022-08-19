/// <https://www.khronos.org/opengl/wiki/Sampler_(GLSL)#Sampler_types>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexDimensionality {
    Tex1d,
    Tex2d,
    Tex3d,
    TexCubeMap,
    TexRectangle,
    Tex1dArray,
    Tex2dArray,
    TexCubeMapArray,
    TexBuffer,
    Tex2dMultisample,
    Tex2dMultisampleArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowSamplerKind {
    Tex1d,
    Tex2d,
    TexCubeMap,
    TexRectangle,
    Tex1dArray,
    Tex2dArray,
    TexCubeMapArray,
}
