use anyhow::Result;
use candle_transformers::models::qwen2::Config as HfConfig;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub model_dir: PathBuf,
    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,
    #[serde(default = "default_max_model_len")]
    pub max_model_len: usize,
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: usize,
    #[serde(default)]
    pub enforce_eager: bool,
    #[serde(default = "default_kvcache_block_size")]
    pub kvcache_block_size: usize,
    #[serde(skip)] // This will be loaded dynamically
    pub hf_config: Option<HfConfig>,
    #[serde(skip)]
    pub eos_token_id: Option<u32>,
    #[serde(skip)]
    pub num_kvcache_blocks: Option<usize>,
}

fn default_max_num_batched_tokens() -> usize { 16384 }
fn default_max_num_seqs() -> usize { 512 }
fn default_max_model_len() -> usize { 4096 }
fn default_gpu_memory_utilization() -> f64 { 0.9 }
fn default_tensor_parallel_size() -> usize { 1 }
fn default_kvcache_block_size() -> usize { 256 }

impl Config {
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
