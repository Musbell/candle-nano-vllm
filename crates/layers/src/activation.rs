#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{Result, Tensor};
use candle_nn::Module;

pub struct SiluAndMul {}

impl SiluAndMul {
    pub fn new() -> Self {
        Self {}
    }
}

impl Module for SiluAndMul {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.rank() - 1;
        let chunks = x.chunk(2, last_dim)?;
        if chunks.len() != 2 {
            candle_core::bail!("expected 2 chunks, got {}", chunks.len());
        }
        let x = &chunks[0];
        let y = &chunks[1];
        x.silu()?.mul(y)
    }
}