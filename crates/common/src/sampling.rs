use serde::Deserialize;


#[derive(Debug, Deserialize)]
pub struct SamplingParams {
    pub temperature: f64,
    pub max_tokens: usize,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: 64,
            ignore_eos: false,
        }
    }
}