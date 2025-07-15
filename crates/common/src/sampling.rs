use serde::{Deserialize, Serialize};


/// Parameters for sampling tokens from the model's output.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct SamplingParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub ignore_eos: bool,
}

// Default values for serde
fn default_temperature() -> f32 { 1.0 }
fn default_max_tokens() -> usize { 1024 }

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            ignore_eos: false,
        }
    }
}
