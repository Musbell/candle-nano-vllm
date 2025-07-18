/// Sequence management for text generation
///
/// This module provides types and functions for managing sequences of tokens
/// during text generation, including tracking their state, handling KV cache
/// blocks, and managing token generation.

use serde::{Deserialize, Serialize};
use std::ops::Index;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::sampling::SamplingParams;

/// Status of a sequence in the generation pipeline
///
/// Represents the current processing state of a sequence as it moves
/// through the text generation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum SequenceStatus {
    /// Sequence is waiting to be processed
    ///
    /// The sequence has been created but has not yet been assigned
    /// to a batch for processing.
    Waiting,
    
    /// Sequence is currently being processed
    ///
    /// The sequence is actively being processed by the model and
    /// may be generating new tokens.
    Running,
    
    /// Sequence has completed processing
    ///
    /// The sequence has finished generating tokens, either because
    /// it reached the maximum length, generated an EOS token, or
    /// was manually stopped.
    Finished,
}

/// Default implementation for SequenceStatus
///
/// Creates a new SequenceStatus with the default value of Waiting,
/// which is the initial state for all new sequences.
impl Default for SequenceStatus {
    fn default() -> Self {
        SequenceStatus::Waiting
    }
}

/// Global counter for generating unique sequence IDs
///
/// This atomic counter ensures that each sequence created during the
/// lifetime of the application gets a unique identifier, even in
/// concurrent environments.
static SEQ_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generates the next unique sequence ID
///
/// Atomically increments the global sequence counter and returns
/// the previous value, ensuring thread-safe generation of unique IDs.
///
/// # Returns
///
/// A unique sequence ID that can be used to identify a sequence.
fn next_seq_id() -> usize {
    SEQ_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Represents a single request/sequence in the text generation system.
///
/// This struct holds the complete state of a sequence, including its token IDs,
/// status, KV cache information, and sampling parameters. It tracks both the
/// original prompt tokens and any generated completion tokens.
///
/// It can be created with `Sequence::new` for new requests or deserialized
/// from a saved state for resuming generation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Sequence {
    // --- Core Identifiers ---
    /// Unique identifier for this sequence
    ///
    /// Generated automatically using the global sequence counter to ensure
    /// each sequence has a unique ID throughout the application's lifetime.
    #[serde(default = "next_seq_id")]
    pub seq_id: usize,
    
    /// Current status of this sequence in the generation pipeline
    ///
    /// Tracks whether the sequence is waiting to be processed, currently
    /// being processed, or has finished processing.
    #[serde(default)]
    pub status: SequenceStatus,

    // --- Token Data ---
    /// All token IDs in this sequence, including both prompt and completion
    ///
    /// This vector contains the token IDs of the original prompt followed by
    /// any tokens that have been generated so far.
    pub token_ids: Vec<u32>,
    
    /// The most recently generated token ID
    ///
    /// This is a cache of the last token in the token_ids vector for
    /// quick access during generation.
    pub last_token_id: u32,
    
    /// Total number of tokens in this sequence
    ///
    /// This is the length of the token_ids vector and includes both
    /// prompt tokens and generated completion tokens.
    pub num_tokens: usize,
    
    /// Number of tokens in the original prompt
    ///
    /// This is used to distinguish between the original prompt tokens
    /// and the generated completion tokens.
    pub num_prompt_tokens: usize,

    // --- KV Cache Management ---
    /// Number of tokens that have been cached in the KV cache
    ///
    /// This tracks how many tokens have had their key-value pairs
    /// computed and stored in the cache.
    #[serde(default)]
    pub num_cached_tokens: usize,
    
    /// The list of physical block numbers in the KV cache
    ///
    /// Maps logical block positions to physical block locations in the
    /// key-value cache, enabling efficient paged attention.
    #[serde(default)]
    pub block_table: Vec<usize>,

    // --- Sampling Parameters ---
    /// Temperature for controlling randomness in token generation
    ///
    /// Higher values (e.g., 1.0) make the output more random,
    /// while lower values (e.g., 0.2) make it more deterministic.
    pub temperature: f32,
    
    /// Maximum number of tokens to generate for this sequence
    ///
    /// The generation process will stop after producing this many tokens,
    /// even if no end-of-sequence token has been generated.
    pub max_tokens: usize,
    
    /// Whether to ignore the end-of-sequence token during generation
    ///
    /// When true, the generation will continue even after an EOS token is produced,
    /// up to the max_tokens limit. When false, generation stops at EOS token.
    pub ignore_eos: bool,
}

impl Sequence {
    /// The size of a block in the KV cache, in tokens
    ///
    /// This constant defines how many tokens are stored in each block
    /// of the key-value cache. It affects memory allocation and efficiency
    /// of the paged attention mechanism.
    pub const BLOCK_SIZE: usize = 256;

    /// Creates a new sequence from a prompt and sampling parameters
    ///
    /// Initializes a new sequence with the provided token IDs as the prompt
    /// and configures it with the specified sampling parameters. The sequence
    /// starts in the Waiting status and has no cached tokens.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Vector of token IDs representing the prompt
    /// * `params` - Sampling parameters to control the generation process
    ///
    /// # Returns
    ///
    /// A new Sequence instance initialized with the provided prompt and parameters
    ///
    /// # Panics
    ///
    /// Panics if `token_ids` is empty, as a sequence must have at least one token
    pub fn new(token_ids: Vec<u32>, params: SamplingParams) -> Self {
        assert!(!token_ids.is_empty(), "Cannot create a sequence with empty token_ids");

        let num_tokens = token_ids.len();

        Self {
            seq_id: next_seq_id(),
            status: SequenceStatus::Waiting,
            // Safe to unwrap due to the assert above.
            last_token_id: *token_ids.last().unwrap(),
            num_prompt_tokens: num_tokens,
            num_tokens,
            token_ids,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            temperature: params.temperature,
            max_tokens: params.max_tokens,
            ignore_eos: params.ignore_eos,
        }
    }

    /// Returns the total number of tokens in the sequence
    ///
    /// This includes both the prompt tokens and any generated completion tokens.
    ///
    /// # Returns
    ///
    /// The total number of tokens in the sequence
    pub fn len(&self) -> usize {
        self.num_tokens
    }

    /// Checks if the sequence is empty
    ///
    /// A sequence is considered empty if it contains no tokens.
    /// Note that this should never be true for a properly initialized sequence,
    /// as the constructor asserts that token_ids is not empty.
    ///
    /// # Returns
    ///
    /// `true` if the sequence has no tokens, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Returns true if the sequence has finished generation
    ///
    /// A sequence is considered finished when its status is set to Finished,
    /// which happens when it reaches the maximum number of tokens, generates
    /// an EOS token (unless ignore_eos is true), or is manually stopped.
    ///
    /// # Returns
    ///
    /// `true` if the sequence has finished generation, `false` otherwise
    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    /// The number of tokens generated by the model, excluding the prompt
    ///
    /// This is calculated as the difference between the total number of tokens
    /// and the number of tokens in the original prompt.
    ///
    /// # Returns
    ///
    /// The number of tokens in the generated completion
    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    /// The token IDs of the prompt
    ///
    /// Returns a slice containing only the token IDs from the original prompt,
    /// excluding any generated completion tokens.
    ///
    /// # Returns
    ///
    /// A slice of the token IDs from the original prompt
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.token_ids[..self.num_prompt_tokens]
    }

    /// The token IDs of the generated completion
    ///
    /// Returns a slice containing only the token IDs that were generated
    /// by the model, excluding the original prompt tokens.
    ///
    /// # Returns
    ///
    /// A slice of the token IDs from the generated completion
    pub fn completion_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_prompt_tokens..]
    }

    /// The number of blocks in the KV cache that are already computed and stored
    ///
    /// This is calculated by dividing the number of cached tokens by the block size.
    ///
    /// # Returns
    ///
    /// The number of complete blocks in the KV cache
    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / Self::BLOCK_SIZE
    }

    /// The total number of blocks required to store the entire sequence
    ///
    /// This calculates how many blocks are needed to store all tokens in the sequence,
    /// rounding up to account for partially filled blocks.
    ///
    /// # Returns
    ///
    /// The total number of blocks needed for the entire sequence
    pub fn num_blocks(&self) -> usize {
        (self.num_tokens + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE
    }

    /// The number of tokens in the last, possibly partially filled, block
    ///
    /// This calculates how many tokens are in the last block of the sequence,
    /// which may not be completely filled.
    ///
    /// # Returns
    ///
    /// The number of tokens in the last block, or 0 if there are no blocks
    pub fn last_block_num_tokens(&self) -> usize {
        let num_blocks = self.num_blocks();
        if num_blocks == 0 {
            0
        } else {
            self.num_tokens - (num_blocks - 1) * Self::BLOCK_SIZE
        }
    }

    /// Returns a slice of token IDs for the i-th block
    ///
    /// Retrieves the token IDs that belong to the specified block index.
    /// Each block contains up to BLOCK_SIZE tokens, except possibly the last block.
    ///
    /// # Arguments
    ///
    /// * `i` - The block index to retrieve
    ///
    /// # Returns
    ///
    /// A slice of token IDs for the specified block
    ///
    /// # Panics
    ///
    /// Panics if the block index is out of bounds (>= num_blocks())
    pub fn block(&self, i: usize) -> &[u32] {
        assert!(i < self.num_blocks(), "Block index out of bounds");
        let start = i * Self::BLOCK_SIZE;
        let end = ((i + 1) * Self::BLOCK_SIZE).min(self.token_ids.len());
        &self.token_ids[start..end]
    }

    /// Appends a new token to the sequence, updating its state
    ///
    /// Adds a new token to the end of the sequence and updates the related
    /// state variables (last_token_id and num_tokens).
    ///
    /// # Arguments
    ///
    /// * `token_id` - The ID of the token to append
    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.last_token_id = token_id;
        self.num_tokens += 1;
    }
}

/// Allows for indexing the sequence's token IDs directly, e.g., `sequence[i]`
///
/// This implementation of the Index trait enables direct access to token IDs
/// using array-like indexing syntax. It provides a convenient way to access
/// individual tokens without having to go through the token_ids field.
///
/// # Examples
///
/// ```
/// let token = sequence[5]; // Gets the 6th token in the sequence
/// ```
///
/// # Panics
///
/// Panics if the index is out of bounds (>= num_tokens).
impl Index<usize> for Sequence {
    /// The type returned by the indexing operation
    type Output = u32;

    /// Returns a reference to the token ID at the specified index
    ///
    /// # Arguments
    ///
    /// * `index` - The position of the token to retrieve
    ///
    /// # Returns
    ///
    /// A reference to the token ID at the specified index
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (>= num_tokens)
    fn index(&self, index: usize) -> &Self::Output {
        &self.token_ids[index]
    }
}