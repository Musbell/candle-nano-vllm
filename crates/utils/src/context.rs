use candle_core::Tensor;
use std::sync::Mutex;

/// Context for model execution
///
/// This struct holds the state and configuration needed for executing
/// large language models efficiently, particularly in a paged attention
/// context. It contains information about sequence lengths, mappings,
/// and other metadata required for both prefill and decode phases.
#[derive(Debug, Clone)]
pub struct Context {
    /// Whether the current execution is in prefill mode
    /// 
    /// When true, the model is processing the initial prompt (prefill phase).
    /// When false, the model is in decode/generation phase.
    pub is_prefill: bool,
    
    /// Cumulative sequence lengths for query
    ///
    /// This tensor contains the cumulative sum of sequence lengths for batched
    /// processing of queries. Used to index into the flattened query tensor.
    pub cu_seqlens_q: Option<Tensor>,
    
    /// Cumulative sequence lengths for key
    ///
    /// This tensor contains the cumulative sum of sequence lengths for batched
    /// processing of keys. Used to index into the flattened key tensor.
    pub cu_seqlens_k: Option<Tensor>,
    
    /// Maximum sequence length for queries
    ///
    /// The length of the longest sequence in the current batch of queries.
    pub max_seqlen_q: usize,
    
    /// Maximum sequence length for keys
    ///
    /// The length of the longest sequence in the current batch of keys.
    pub max_seqlen_k: usize,
    
    /// Slot mapping tensor
    ///
    /// Maps token positions to their corresponding memory locations in the KV cache.
    pub slot_mapping: Option<Tensor>,
    
    /// Context lengths tensor
    ///
    /// Contains the length of context for each sequence in the batch.
    pub context_lens: Option<Tensor>,
    
    /// Block tables for paged attention
    ///
    /// Contains the mapping of logical blocks to physical blocks in memory
    /// for efficient paged key-value cache implementation.
    pub block_tables: Option<Vec<Tensor>>
}

/// Default implementation for Context
///
/// Creates a new Context with default values:
/// - is_prefill: false (decode mode by default)
/// - All tensor fields set to None
/// - max_seqlen_q and max_seqlen_k set to 0
impl Default for Context {
    fn default() -> Self {
        Self {
            is_prefill: false,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping: None,
            context_lens: None,
            block_tables: None
        }
    }
}

impl Context {
    /// Creates a new Context with default values
    ///
    /// This is a convenience method that calls Default::default()
    /// to create a new Context instance with all default values.
    pub fn new() -> Self { Self::default() }
}

/// Global context instance protected by a mutex
///
/// This static variable holds the global context that can be accessed
/// and modified by the get_context() and set_context() functions.
/// It's wrapped in Option<T> to allow for the case where no context
/// has been set yet.
static CONTEXT: Mutex<Option<Context>> = Mutex::new(None);

/// Get the current global context
///
/// Returns a clone of the current global context if one has been set,
/// or a new default context if none has been set yet.
///
/// # Returns
///
/// A `Context` instance that is either a clone of the current global context
/// or a new default context.
///
/// # Thread Safety
///
/// This function acquires a lock on the global context mutex, ensuring
/// thread-safe access to the global context.
pub fn get_context() -> Context {
    let context = CONTEXT.lock().unwrap();
    match &*context {
        Some(ctx) => ctx.clone(),
        None => Context::default(),
    }
}

/// Set the global context with new values
///
/// Updates the global context with the provided values. This function
/// is used to configure the execution environment for model operations.
///
/// # Arguments
///
/// * `is_prefill` - Whether the current execution is in prefill mode
/// * `cu_seqlens_q` - Cumulative sequence lengths for query
/// * `cu_seqlens_k` - Cumulative sequence lengths for key
/// * `max_seqlen_q` - Maximum sequence length for queries
/// * `max_seqlen_k` - Maximum sequence length for keys
/// * `slot_mapping` - Maps token positions to their corresponding memory locations
/// * `context_lens` - Contains the length of context for each sequence
/// * `block_tables` - Contains the mapping of logical blocks to physical blocks
///
/// # Thread Safety
///
/// This function acquires a lock on the global context mutex, ensuring
/// thread-safe modification of the global context.
pub fn set_context(
    is_prefill: bool,
    cu_seqlens_q: Option<Tensor>,
    cu_seqlens_k: Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    slot_mapping: Option<Tensor>,
    context_lens: Option<Tensor>,
    block_tables: Option<Vec<Tensor>>,
) {
    let mut context = CONTEXT.lock().unwrap();
    *context = Some(Context {
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    });
}