/// Utility module for the candle-nano-vllm project
///
/// This crate provides various utility functions and data structures
/// used throughout the project, including context management for model execution
/// and weight loading from safetensors files.

mod context;
mod loader;

/// Re-exports from the context module
///
/// These exports provide access to the Context struct and related functions
/// for managing the global execution context in the model.
pub use context::{Context, get_context, set_context};

/// Re-exports from the loader module
///
/// These exports provide functionality for loading weights from safetensors files
/// into candle-based models.
pub use loader::{SafeTensorLoadable, PackedModulesMapping, load_model};

/// Simple utility function that adds two numbers
///
/// # Arguments
///
/// * `left` - First number to add
/// * `right` - Second number to add
///
/// # Returns
///
/// The sum of `left` and `right`
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
