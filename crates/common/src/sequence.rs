use crate::sampling::SamplingParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
     Waiting,
    Running,
    Finished,
}

// Represents a single request/sequence in the system.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub seq_id:usize,
    pub status: SequenceStatus,
    pub token_ids: Vec<u32>,
    pub last_tokens: usize,
    pub num_tokens: usize,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub temperature: SequenceStatus,
    pub max_tokens: SequenceStatus,
    pub ignore_eos: SequenceStatus,
    // The list of physical block numbers in the KV cache
    pub block_table: Vec<usize>,
}