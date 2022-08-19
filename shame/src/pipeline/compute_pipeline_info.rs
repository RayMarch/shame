//! pipeline info for creating compute pipeline layouts
use std::fmt::Display;

use super::render_pipeline_info::{BindGroupInfo, PushConstantInfo};

/// Additional info to the recorded compute shader, which is necessary to create
/// a compute pipeline.
///
/// Members which are `None` or empty have not been recorded
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ComputePipelineInfo {
    /// 3d dimensions of each work group in this compute pipeline
    pub work_group_size: Option<[usize; 3]>,
    /// bind group layout information of all attached bind groups
    pub bind_groups: Vec<BindGroupInfo>,
    /// push constant format
    pub push_constant: Option<PushConstantInfo>,
}

impl Display for ComputePipelineInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        {
            let [x, y, z] = self.work_group_size.unwrap_or([1, 1, 1]);
            f.write_fmt(format_args!("work group size: [{}, {}, {}]\n", x, y, z))?;
        }

        match &self.push_constant {
            None => f.write_str("no push constant\n")?,
            Some(x) => f.write_fmt(format_args!("push constant: {x}\n"))?,
        }
        match &self.bind_groups[..] {
            [] => f.write_str("no bind groups")?,
            s => {
                f.write_fmt(format_args!("bind groups:\n"))?;
                for x in s {
                    f.write_fmt(format_args!("{x}"))?;
                }
            }
        }
        Ok(())
    }
}
