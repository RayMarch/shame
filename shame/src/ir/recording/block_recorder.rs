use super::{Block, BlockError, BlockKind, CallInfo, Context};
use crate::{
    call_info,
    common::{pool::Key, small_vec::SmallVec},
};

pub struct BlockSeriesRecorder<const N: usize> {
    initial_caller: CallInfo,
    error: ErrorState,
    /// the common parent block of all blocks in the series
    parent: Option<Key<Block>>,
    expected_current: Option<Key<Block>>,
    blocks: SmallVec<Key<Block>, N>,
    closed_gracefully: bool,
}

impl<const N: usize> Drop for BlockSeriesRecorder<N> {
    #[track_caller]
    fn drop(&mut self) {
        if !self.closed_gracefully {
            Context::try_with(call_info!(), |ctx| {
                ctx.push_error(BlockError::UnfinishedBlockRecording.into())
            });
        }
    }
}

impl<const N: usize> BlockSeriesRecorder<N> {
    pub fn new(call_info: CallInfo, kind: BlockKind) -> Self {
        Context::try_with(call_info, |ctx| {
            if kind.has_different_execution_state() {
                ctx.increment_execution_state();
            }
            // create a block on top of the current block
            let mut new_block = Block::new_with_parent(ctx, kind, ctx.current_block());
            // we expect the current block to be that block once we replace it
            let expected_current = Some(new_block);
            // replace the current block with this block
            let parent = ctx.replace_current_block(new_block);
            Self {
                initial_caller: call_info,
                error: ErrorState::new_ok(),
                parent: Some(parent),
                expected_current,
                blocks: Default::default(),
                closed_gracefully: false,
            }
        })
        .unwrap_or_else(|| Self {
            initial_caller: call_info,
            error: BlockError::BlockOutsideOfEncoding.into(),
            parent: None,
            expected_current: None,
            blocks: Default::default(),
            closed_gracefully: false,
        })
    }

    pub fn initial_caller(&self) -> CallInfo { self.initial_caller }

    /// ends the current block and starts the next block in the chain
    pub fn advance(&mut self, call_info: CallInfo, kind: BlockKind) {
        Context::try_with(call_info, |ctx| match (self.parent, &mut self.expected_current) {
            (Some(parent), Some(expected_current)) => {
                let new_block = Block::new_with_parent(ctx, kind, parent);
                let completed_block = ctx.replace_current_block(new_block);

                let completed_kind = ctx.pool()[completed_block].kind;
                if completed_kind.has_different_execution_state() || kind.has_different_execution_state() {
                    ctx.increment_execution_state();
                }

                if *expected_current != completed_block {
                    ctx.push_error(BlockError::UnfinishedBlockRecording.into())
                }
                *expected_current = new_block;

                self.blocks.push(completed_block)
            }
            _ => ctx.push_error(BlockError::IllFormedBlockSeriesRecorder.into()),
        })
        .unwrap_or_else(|| {
            // this information can no longer be used.
            // since the context is gone, we can't even use it to make
            // nicer error messages, therefore we just clear it altogether
            *self = Self {
                initial_caller: self.initial_caller,
                error: BlockError::BlockOutsideOfEncoding.into(),
                parent: None,
                expected_current: None,
                blocks: Default::default(),
                closed_gracefully: false,
            };
        })
    }

    pub fn finish<const K: usize>(mut self, call_info: CallInfo) -> Result<[Key<Block>; K], BlockError> {
        self.closed_gracefully = true;
        Context::try_with(call_info, |ctx| {
            self.error.push_if_needed(ctx);
            match (self.parent, &mut self.expected_current) {
                (Some(parent), Some(expected_current)) => {
                    let completed_block = ctx.replace_current_block(parent);

                    let completed_kind = ctx.pool()[completed_block].kind;
                    if completed_kind.has_different_execution_state() {
                        ctx.increment_execution_state();
                    }

                    if *expected_current != completed_block {
                        ctx.push_error(BlockError::UnfinishedBlockRecording.into())
                    }
                    self.expected_current = None;
                    self.blocks.push(completed_block);

                    let blocks = std::mem::take(&mut self.blocks);
                    <[Key<Block>; K]>::try_from(blocks).map_err(|e| BlockError::UnexpectedAmountOfBlocks {
                        expected: K,
                        actual: e.len(),
                    })
                }
                _ => Err(BlockError::IllFormedBlockSeriesRecorder),
            }
        })
        .unwrap_or(Err(BlockError::BlockOutsideOfEncoding))
    }
}

struct ErrorState {
    error: Option<BlockError>,
    /// whether the error was successfully pushed to the context yet.
    /// this is used to prevent reporting an error twice
    pushed: bool,
}

impl From<BlockError> for ErrorState {
    fn from(error: BlockError) -> Self {
        Self {
            error: Some(error),
            pushed: false,
        }
    }
}

impl ErrorState {
    fn is_ok(&self) -> bool { self.error.is_some() }

    fn new_ok() -> Self {
        Self {
            error: None,
            pushed: false,
        }
    }

    fn push_if_needed(&mut self, ctx: &Context) {
        if let Some(error) = self.error.clone() {
            if std::mem::replace(&mut self.pushed, true) {
                ctx.push_error(error.into())
            }
        }
    }
}
