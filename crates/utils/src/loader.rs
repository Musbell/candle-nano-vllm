/// Weight loading utilities for candle-based models
///
/// This module provides functionality for loading weights from safetensors files
/// into candle-based models. It supports loading weights for both standard models
/// and models with packed modules (where weights are split across multiple tensors).
use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, Context as _};
use candle_core::{DType, Device, Tensor};
use glob::glob;
use safetensors::SafeTensors;
use std::fs;

/// Trait for models that can load weights from safetensors files
///
/// This trait defines the interface for models that can load weights from
/// safetensors files. It provides methods for loading weights by name and
/// handling packed modules (where weights are split across multiple tensors).
///
/// Implementing this trait allows a model to be used with the `load_model` function,
/// which loads weights from safetensors files into the model.
pub trait SafeTensorLoadable {
    /// Get the packed modules mapping for this model
    ///
    /// Packed modules are used when a single logical weight is split across multiple
    /// tensors, such as in sharded models. The mapping associates weight name patterns
    /// with replacement patterns and shard IDs.
    ///
    /// # Returns
    ///
    /// An Option containing a reference to the packed modules mapping if it exists,
    /// or None if the model doesn't use packed modules.
    ///
    /// The mapping is from a weight name pattern (String) to a tuple of:
    /// - Replacement pattern (String): Used to transform the weight name
    /// - Shard ID (usize): Identifies which shard of the weight this is
    fn get_packed_modules_mapping(&self) -> Option<&HashMap<String, (String, usize)>> {
        None
    }
    
    /// Load a weight tensor into a parameter
    ///
    /// This method is called for each weight tensor found in the safetensors files.
    /// It should find the parameter with the given name in the model and update it
    /// with the provided weight tensor.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter to load the weight into
    /// * `weight` - The weight tensor to load
    /// * `shard_id` - Optional shard ID for packed modules, indicating which shard
    ///                of a split weight this tensor represents
    ///
    /// # Returns
    ///
    /// Result indicating success or an error:
    /// - `Ok(true)` if the parameter was found and the weight was successfully loaded
    /// - `Ok(false)` if the parameter was not found in the model
    /// - `Err(...)` if there was an error loading the weight
    ///
    /// # Implementation Notes
    ///
    /// For packed modules, implementations should use the shard_id to determine
    /// how to apply the weight to the parameter. For example, different shards
    /// might need to be concatenated or applied to different parts of the parameter.
    fn load_weight(&mut self, name: &str, weight: Tensor, shard_id: Option<usize>) -> Result<bool>;
}

/// Type for packed module mapping
/// 
/// Maps from a weight name pattern to a tuple of (replacement pattern, shard_id).
/// This is used to handle cases where a single logical weight is split across
/// multiple tensors, such as in sharded models.
pub type PackedModulesMapping = HashMap<String, (String, usize)>;

/// Convert a safetensors dtype to a candle-core DType
///
/// # Arguments
///
/// * `dtype` - The safetensors dtype to convert
/// * `tensor_name` - The name of the tensor (used for error messages)
///
/// # Returns
///
/// The corresponding candle-core DType
///
/// # Errors
///
/// Returns an error if the dtype is not supported
fn convert_dtype(dtype: safetensors::tensor::Dtype, tensor_name: &str) -> Result<DType> {
    match dtype {
        safetensors::tensor::Dtype::F32 => Ok(DType::F32),
        safetensors::tensor::Dtype::F16 => Ok(DType::F16),
        safetensors::tensor::Dtype::BF16 => Ok(DType::BF16),
        safetensors::tensor::Dtype::I64 => Ok(DType::I64),
        safetensors::tensor::Dtype::I32 => Ok(DType::U32), // Map I32 to U32
        safetensors::tensor::Dtype::U8 => Ok(DType::U8),
        safetensors::tensor::Dtype::I8 => Ok(DType::U8),   // Map I8 to U8
        safetensors::tensor::Dtype::BOOL => Ok(DType::U8), // Map BOOL to U8
        _ => anyhow::bail!("Unsupported dtype for tensor {}", tensor_name),
    }
}

/// Create a tensor from safetensors data
///
/// # Arguments
///
/// * `view` - The safetensors tensor view
/// * `tensor_name` - The name of the tensor (used for error messages)
///
/// # Returns
///
/// A candle-core Tensor
///
/// # Errors
///
/// Returns an error if:
/// - The dtype is not supported
/// - The tensor cannot be created from the data
fn create_tensor(view: &impl safetensors::tensor::View, tensor_name: &str) -> Result<Tensor> {
    let shape = view.shape().to_vec();
    let dtype = convert_dtype(view.dtype(), tensor_name)?;
    
    Ok(Tensor::from_raw_buffer(
        &view.data(),
        dtype,
        &shape,
        &Device::Cpu,
    )?)
}

/// Find a matching packed module mapping for a tensor name
///
/// # Arguments
///
/// * `tensor_name` - The name of the tensor to find a mapping for
/// * `mapping` - The packed modules mapping to search in
///
/// # Returns
///
/// If a matching mapping is found, returns a tuple containing:
/// - The parameter name (with the pattern replaced)
/// - The shard ID
/// Otherwise, returns None
fn find_packed_mapping(tensor_name: &str, mapping: &PackedModulesMapping) -> Option<(String, usize)> {
    for (pattern, (replacement, shard_id)) in mapping {
        if tensor_name.contains(pattern) {
            let param_name = tensor_name.replace(pattern, replacement);
            return Some((param_name, *shard_id));
        }
    }
    None
}

/// Process a single tensor from a safetensors file
///
/// # Arguments
///
/// * `model` - The model to load the weight into
/// * `tensors` - The safetensors file
/// * `tensor_name` - The name of the tensor to process
/// * `packed_modules_mapping` - Optional mapping for packed modules
///
/// # Returns
///
/// Result indicating success or an error
///
/// # Errors
///
/// Returns an error if:
/// - The tensor cannot be retrieved from the safetensors file
/// - The tensor cannot be converted to a candle-core Tensor
/// - The model's `load_weight` method returns an error
fn process_tensor<M: SafeTensorLoadable>(
    model: &mut M,
    tensors: &SafeTensors,
    tensor_name: &str,
    packed_modules_mapping: &Option<PackedModulesMapping>,
) -> Result<()> {
    // Check if this weight is part of a packed module
    let (param_name, shard_id) = if let Some(mapping) = packed_modules_mapping {
        if let Some((name, id)) = find_packed_mapping(tensor_name, mapping) {
            (name, Some(id))
        } else {
            (tensor_name.to_string(), None)
        }
    } else {
        (tensor_name.to_string(), None)
    };
    
    // Get the tensor data and create a candle-core Tensor
    let view = tensors.tensor(tensor_name)?;
    let tensor = create_tensor(&view, tensor_name)?;
    
    // Load the weight into the parameter
    if !model.load_weight(&param_name, tensor, shard_id)? {
        // Parameter not found, log a warning
        eprintln!("Warning: Parameter {} not found in model", param_name);
    }
    
    Ok(())
}

/// Load model weights from safetensors files
///
/// This function loads weights from safetensors files into a model that implements
/// the `SafeTensorLoadable` trait. It handles both standard weights and packed
/// modules (where weights are split across multiple tensors).
///
/// # Arguments
///
/// * `model` - The model to load weights into, must implement `SafeTensorLoadable`
/// * `path` - Path to the directory containing safetensors files
///
/// # Returns
///
/// Result indicating success or an error
///
/// # Error Handling
///
/// This function will return an error if:
/// - The path doesn't exist or can't be read
/// - The safetensors files can't be parsed
/// - There's an error creating tensors from the safetensors data
/// - The model's `load_weight` method returns an error
///
/// # Notes
///
/// - This function will log warnings for parameters that are in the safetensors
///   files but not found in the model.
/// - It automatically handles data type conversions from safetensors types to
///   candle-core types.
pub fn load_model<M: SafeTensorLoadable>(
    model: &mut M,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    let pattern = path.join("*.safetensors");
    let pattern_str = pattern.to_string_lossy();
    
    // Get the packed modules mapping if available
    let packed_modules_mapping = model.get_packed_modules_mapping().cloned();
    
    // Find all safetensors files in the directory
    for entry in glob(&pattern_str)
        .with_context(|| format!("Failed to read glob pattern {}", pattern_str))?
    {
        let file_path = entry?;
        let data = fs::read(&file_path)
            .with_context(|| format!("Failed to read file {}", file_path.display()))?;
        
        // Open the safetensors file
        let tensors = SafeTensors::deserialize(&data)?;
        
        // Process each weight in the file
        for tensor_name in tensors.names() {
            process_tensor(model, &tensors, tensor_name, &packed_modules_mapping)?;
        }
    }
    
    Ok(())
}