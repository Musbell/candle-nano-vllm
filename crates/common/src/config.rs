/// Configuration for model loading and inference
///
/// This module provides types and functions for configuring the behavior
/// of language models, including memory usage, batch sizes, and other
/// performance-related parameters.

use anyhow::Result;
use candle_transformers::models::qwen2::Config as HfConfig;
use serde::Deserialize;
use std::path::PathBuf;

/// Configuration for model loading and inference
///
/// This struct contains all the configuration parameters needed to load
/// and run a language model efficiently. It includes settings for memory
/// management, parallelism, and model-specific parameters.
///
/// The configuration can be loaded from a file or created programmatically.
/// Many fields have sensible defaults that can be overridden as needed.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    /// Directory containing the model files
    ///
    /// This should point to a directory containing the model weights,
    /// configuration files, and any other necessary assets.
    #[serde(default)]
    pub model_dir: PathBuf,
    
    /// Maximum number of tokens to process in a single batch
    ///
    /// This limits the total number of tokens that can be processed
    /// simultaneously across all sequences in a batch. Higher values
    /// increase throughput but require more memory.
    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,
    
    /// Maximum number of sequences to process in a single batch
    ///
    /// This limits the total number of sequences that can be processed
    /// simultaneously. Higher values increase throughput for many short
    /// sequences but require more memory.
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,
    
    /// Maximum sequence length supported by the model
    ///
    /// This is the maximum number of tokens that can be in a single sequence,
    /// including both prompt and generated tokens. It's typically determined
    /// by the model's training configuration.
    #[serde(default = "default_max_model_len")]
    pub max_model_len: usize,
    
    /// Fraction of GPU memory to use for the model
    ///
    /// This controls how much of the available GPU memory will be allocated
    /// for model weights and the KV cache. Values closer to 1.0 use more
    /// memory but may improve performance.
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,
    
    /// Number of GPUs to use for tensor parallelism
    ///
    /// When greater than 1, the model will be split across multiple GPUs
    /// using tensor parallelism. This can improve performance for large models
    /// but requires multiple GPUs.
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: usize,
    
    /// Whether to enforce eager execution (disable CUDA graphs)
    ///
    /// When true, CUDA graphs will not be used even if available.
    /// This can be useful for debugging or in environments where
    /// CUDA graphs cause issues.
    #[serde(default)]
    pub enforce_eager: bool,
    
    /// Size of each block in the KV cache, in tokens
    ///
    /// This controls the granularity of memory allocation in the paged
    /// attention mechanism. Larger blocks may improve performance but
    /// can waste memory for short sequences.
    #[serde(default = "default_kvcache_block_size")]
    pub kvcache_block_size: usize,
    
    /// Hugging Face model configuration
    ///
    /// This contains the model-specific parameters loaded from the
    /// config.json file in the model directory. It's loaded dynamically
    /// and not deserialized directly from configuration files.
    #[serde(skip)] // This will be loaded dynamically
    pub hf_config: Option<HfConfig>,
    
    /// End-of-sequence token ID for the model
    ///
    /// This is the token ID that indicates the end of a sequence.
    /// It's typically loaded from the model's tokenizer configuration.
    #[serde(skip)]
    pub eos_token_id: Option<u32>,
    
    /// Number of blocks to allocate for the KV cache
    ///
    /// This is calculated based on available memory and other configuration
    /// parameters. It determines how many blocks will be allocated for
    /// the key-value cache.
    #[serde(skip)]
    pub num_kvcache_blocks: Option<usize>,
}

/// Default value for maximum number of tokens in a batch
///
/// Returns 16384, which provides a good balance between throughput
/// and memory usage for most hardware configurations.
fn default_max_num_batched_tokens() -> usize { 16384 }

/// Default value for maximum number of sequences in a batch
///
/// Returns 512, which allows for efficient batching of multiple
/// sequences while keeping memory requirements reasonable.
fn default_max_num_seqs() -> usize { 512 }

/// Default value for maximum model sequence length
///
/// Returns 4096, which is a common context window size for
/// many modern language models.
fn default_max_model_len() -> usize { 4096 }

/// Default value for GPU memory utilization
///
/// Returns 0.9 (90%), which reserves some GPU memory for system
/// operations while maximizing the memory available for the model.
fn default_gpu_memory_utilization() -> f64 { 0.9 }

/// Default value for tensor parallel size
///
/// Returns 1, which means no tensor parallelism by default.
/// This is appropriate for single-GPU setups.
fn default_tensor_parallel_size() -> usize { 1 }

/// Default value for KV cache block size
///
/// Returns 256 tokens per block, which provides a good balance
/// between memory efficiency and performance for most use cases.
fn default_kvcache_block_size() -> usize { 256 }

impl Config {
    /// Creates a new Config from a model directory
    ///
    /// This constructor loads the Hugging Face model configuration from
    /// the specified directory and initializes a new Config with default
    /// values for all other fields.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to the directory containing the model files
    ///
    /// # Returns
    ///
    /// A Result containing the new Config if successful, or an error if
    /// the model configuration could not be loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The config.json file does not exist in the model directory
    /// - The file cannot be read
    /// - The file contains invalid JSON
    /// - The JSON does not match the expected HfConfig structure
    pub fn new(model_dir: PathBuf) -> Result<Self> {
        // TODO: Load from a file, but for now, we construct it.
        let hf_config_path = model_dir.join("config.json");
        let hf_config: HfConfig = serde_json::from_str(&std::fs::read_to_string(hf_config_path)?)?;

        Ok(Self {
            model_dir,
            hf_config: Some(hf_config),
            ..Default::default()
        })
    }
}
