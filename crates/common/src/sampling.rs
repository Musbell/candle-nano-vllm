use serde::{Deserialize, Serialize};

/// Parameters for sampling tokens from the model's output.
///
/// This struct contains configuration parameters that control how tokens
/// are sampled from the model's output distribution during text generation.
/// It allows customization of the generation process through temperature,
/// maximum token count, and end-of-sequence handling.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct SamplingParams {
    /// Temperature for controlling randomness in sampling
    ///
    /// Higher values (e.g., 1.0) make the output more random,
    /// while lower values (e.g., 0.2) make it more deterministic.
    /// A value of 0.0 will result in greedy sampling (always selecting the most likely token).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    
    /// Maximum number of tokens to generate
    ///
    /// This limits the total length of the generated sequence.
    /// The generation process will stop after producing this many tokens,
    /// even if no end-of-sequence token has been generated.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    
    /// Whether to ignore the end-of-sequence token during generation
    ///
    /// When true, the generation will continue even after an EOS token is produced,
    /// up to the max_tokens limit. When false, generation stops at EOS token.
    #[serde(default)]
    pub ignore_eos: bool,
}

/// Default temperature value for token sampling
///
/// Returns 1.0, which provides a balanced level of randomness in generation.
/// This is used as the default value for the temperature field in SamplingParams.
fn default_temperature() -> f32 { 1.0 }

/// Default maximum number of tokens to generate
///
/// Returns 1024, which is a reasonable limit for most generation tasks.
/// This is used as the default value for the max_tokens field in SamplingParams.
fn default_max_tokens() -> usize { 1024 }

/// Default implementation for SamplingParams
///
/// Creates a new SamplingParams instance with default values:
/// - temperature: 1.0 (balanced randomness)
/// - max_tokens: 1024 (reasonable generation limit)
/// - ignore_eos: false (generation stops at end-of-sequence token)
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            ignore_eos: false,
        }
    }
}
