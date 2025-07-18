/// Activation functions for neural network layers
///
/// This module provides implementations of various activation functions
/// used in transformer-based language models, particularly those that
/// involve specialized operations beyond standard activations.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{Result, Tensor};
#[cfg(feature = "candle-nn")]
use candle_nn::Module;

/// SiLU activation followed by multiplication with a gating function
///
/// This activation function is commonly used in modern transformer architectures
/// like LLaMA, Mistral, and others. It implements the SwiGLU-style activation
/// where:
/// 1. The input tensor is split into two equal parts along the last dimension
/// 2. The first part has SiLU activation applied (x * sigmoid(x))
/// 3. The result is multiplied element-wise with the second part
///
/// This is a key component in gated feed-forward networks.
pub struct SiluAndMul {}

impl SiluAndMul {
    /// Creates a new SiluAndMul activation function
    ///
    /// This constructor takes no parameters as the activation function
    /// has no learnable parameters or configuration options.
    ///
    /// # Returns
    ///
    /// A new instance of the SiluAndMul activation function
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(feature = "candle-nn")]
impl Module for SiluAndMul {
    /// Applies the SiluAndMul activation function to the input tensor
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor that will be split into two parts along the last dimension.
    ///         The tensor must have an even size in its last dimension.
    ///
    /// # Returns
    ///
    /// A tensor with half the size of the input in the last dimension, where each element
    /// is the result of SiLU(x₁) * x₂, where x₁ and x₂ are the corresponding elements
    /// from the first and second halves of the input.
    ///
    /// # Errors
    ///
    /// Returns an error if the input tensor cannot be split into exactly two chunks
    /// along the last dimension.
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

/// Implementation of the SiluAndMul forward pass without the Module trait
///
/// This function provides the core functionality of SiluAndMul even when
/// the candle-nn feature is not enabled.
impl SiluAndMul {
    /// Applies the SiluAndMul activation function to the input tensor
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor that will be split into two parts along the last dimension.
    ///         The tensor must have an even size in its last dimension.
    ///
    /// # Returns
    ///
    /// A tensor with half the size of the input in the last dimension, where each element
    /// is the result of SiLU(x₁) * x₂, where x₁ and x₂ are the corresponding elements
    /// from the first and second halves of the input.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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