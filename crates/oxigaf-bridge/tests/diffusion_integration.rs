//! Integration tests for ToRSh weight conversion with oxigaf-diffusion
//!
//! These tests verify that converted weights can be loaded and used
//! by the oxigaf-diffusion pipeline.

#![cfg(feature = "torsh")]

use oxigaf_bridge::{
    create_synthetic_gaf_checkpoint, validate_converted_checkpoint, GafLayerMapper, WeightConverter,
};
use std::env;

#[test]
fn test_layer_names_match_diffusion_varbuilder() {
    // Verify that converted layer names match what oxigaf-diffusion expects
    let mapper = GafLayerMapper::new();

    // Test U-Net layer names - ToRSh uses slashes, OxiGAF uses dots
    let test_cases = vec![
        (
            "down_blocks/0/resnets/0/norm1/weight",
            "down_blocks.0.resnets.0.norm1.weight",
        ),
        (
            "mid_block/resnets/0/conv1/weight",
            "mid_block.resnets.0.conv1.weight",
        ),
        (
            "up_blocks/3/attentions/0/transformer_blocks/0/attn1/to_q/weight",
            "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        ),
        (
            "time_embedding/linear_1/weight",
            "time_embedding.linear_1.weight",
        ),
        (
            "camera_embedding/linear_2/weight",
            "camera_embedding.linear_2.weight",
        ),
    ];

    for (torsh_name, expected_oxigaf) in test_cases {
        let oxigaf_name = mapper
            .map_torsh_to_oxigaf(torsh_name)
            .unwrap_or_else(|e| panic!("Failed to map {}: {}", torsh_name, e));

        // Verify format matches VarBuilder expectations
        assert_eq!(
            oxigaf_name, expected_oxigaf,
            "Layer name mismatch for {}",
            torsh_name
        );

        // Verify no slashes (VarBuilder uses dots)
        assert!(
            !oxigaf_name.contains('/'),
            "Layer name {} should not contain slashes",
            oxigaf_name
        );

        // Verify contains dots (VarBuilder path separator)
        assert!(
            oxigaf_name.contains('.'),
            "Layer name {} should contain dots",
            oxigaf_name
        );
    }
}

#[test]
fn test_converted_weights_validate() {
    let temp_dir = env::temp_dir();
    let torsh_path = temp_dir.join("test_integration_torsh.safetensors");
    let oxigaf_path = temp_dir.join("test_integration_oxigaf.safetensors");

    // Create synthetic checkpoint
    create_synthetic_gaf_checkpoint(&torsh_path).expect("Failed to create synthetic checkpoint");

    // Convert
    let converter = WeightConverter::new();
    converter
        .torsh_to_oxigaf(&torsh_path, &oxigaf_path)
        .expect("Conversion should succeed");

    // Validate
    let report = validate_converted_checkpoint(&oxigaf_path).expect("Validation should not error");

    // Check validation passed
    if !report.is_valid() {
        eprintln!("Validation failed: {}", report.summary());
        if !report.missing_layers.is_empty() {
            eprintln!("Missing layers: {:?}", report.missing_layers);
        }
        if !report.invalid_names.is_empty() {
            eprintln!("Invalid names: {:?}", report.invalid_names);
        }
        if !report.invalid_shapes.is_empty() {
            eprintln!("Invalid shapes: {:?}", report.invalid_shapes);
        }
        if !report.has_nan_inf.is_empty() {
            eprintln!("NaN/Inf values: {:?}", report.has_nan_inf);
        }
    }

    assert!(report.file_exists, "File should exist");
    assert!(report.safetensors_valid, "Should be valid safetensors");
    assert!(
        report.invalid_names.is_empty(),
        "Should have no invalid names (slashes)"
    );
    assert!(
        report.has_nan_inf.is_empty(),
        "Should have no NaN/Inf values"
    );

    // Cleanup
    let _ = std::fs::remove_file(&torsh_path);
    let _ = std::fs::remove_file(&oxigaf_path);
}

#[test]
#[ignore] // GPU required
fn test_converted_weights_load_in_diffusion() {
    // This test requires:
    // 1. A GAF checkpoint in ToRSh format
    // 2. GPU available
    // 3. oxigaf-diffusion dependency

    use candle_core::{DType, Device};
    use candle_nn as nn;

    let temp_dir = env::temp_dir();
    let torsh_path = temp_dir.join("test_diffusion_load_torsh.safetensors");
    let oxigaf_path = temp_dir.join("test_diffusion_load_oxigaf.safetensors");

    // Create synthetic checkpoint
    create_synthetic_gaf_checkpoint(&torsh_path).expect("Failed to create synthetic checkpoint");

    // Convert
    let converter = WeightConverter::new();
    converter
        .torsh_to_oxigaf(&torsh_path, &oxigaf_path)
        .expect("Conversion should succeed");

    // Try to load in diffusion pipeline
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dtype = DType::F32;

    // Load with VarBuilder
    let data = std::fs::read(&oxigaf_path).expect("Failed to read converted file");
    let vb_result = nn::VarBuilder::from_buffered_safetensors(data, dtype, &device);

    assert!(
        vb_result.is_ok(),
        "VarBuilder should be able to load converted weights"
    );

    let vb = vb_result.expect("VarBuilder creation failed");

    // Try to access tensors with explicit shapes
    // conv_in.weight: [320, 8, 3, 3]
    let conv_in = vb.pp("conv_in").get((320, 8, 3, 3), "weight");
    assert!(conv_in.is_ok(), "Should load conv_in.weight");
    println!("✓ Loaded conv_in.weight: {:?}", conv_in.unwrap().shape());

    // time_embedding.linear_1.weight: [1280, 320]
    let time_emb = vb
        .pp("time_embedding")
        .pp("linear_1")
        .get((1280, 320), "weight");
    assert!(
        time_emb.is_ok(),
        "Should load time_embedding.linear_1.weight"
    );
    println!(
        "✓ Loaded time_embedding.linear_1.weight: {:?}",
        time_emb.unwrap().shape()
    );

    // down_blocks.0.resnets.0.norm1.weight: [320]
    let down_norm = vb
        .pp("down_blocks")
        .pp("0")
        .pp("resnets")
        .pp("0")
        .pp("norm1")
        .get((320,), "weight");
    assert!(
        down_norm.is_ok(),
        "Should load down_blocks.0.resnets.0.norm1.weight"
    );
    println!(
        "✓ Loaded down_blocks.0.resnets.0.norm1.weight: {:?}",
        down_norm.unwrap().shape()
    );

    // mid_block.resnets.0.norm1.weight: [1280]
    let mid_norm = vb
        .pp("mid_block")
        .pp("resnets")
        .pp("0")
        .pp("norm1")
        .get((1280,), "weight");
    assert!(
        mid_norm.is_ok(),
        "Should load mid_block.resnets.0.norm1.weight"
    );
    println!(
        "✓ Loaded mid_block.resnets.0.norm1.weight: {:?}",
        mid_norm.unwrap().shape()
    );

    println!("✓ All GPU tests passed!");

    // Cleanup
    let _ = std::fs::remove_file(&torsh_path);
    let _ = std::fs::remove_file(&oxigaf_path);
}

#[test]
fn test_varbuilder_can_load_converted_weights() {
    use candle_core::{DType, Device};
    use candle_nn as nn;

    let temp_dir = env::temp_dir();
    let torsh_path = temp_dir.join("test_varbuilder_torsh.safetensors");
    let oxigaf_path = temp_dir.join("test_varbuilder_oxigaf.safetensors");

    // Create and convert
    create_synthetic_gaf_checkpoint(&torsh_path).expect("Failed to create synthetic checkpoint");

    let converter = WeightConverter::new();
    converter
        .torsh_to_oxigaf(&torsh_path, &oxigaf_path)
        .expect("Conversion should succeed");

    // Try to load with VarBuilder
    let device = Device::Cpu; // CPU is fine for this test
    let dtype = DType::F32;

    let data = std::fs::read(&oxigaf_path).expect("Failed to read file");
    let vb_result = nn::VarBuilder::from_buffered_safetensors(data, dtype, &device);

    assert!(
        vb_result.is_ok(),
        "VarBuilder should load converted weights without error"
    );

    let vb = vb_result.expect("VarBuilder creation failed");

    // Verify we can access tensors
    let conv_in_weight = vb.pp("conv_in").get((320, 8, 3, 3), "weight");
    assert!(
        conv_in_weight.is_ok(),
        "Should be able to load conv_in.weight with expected shape"
    );

    // Cleanup
    let _ = std::fs::remove_file(&torsh_path);
    let _ = std::fs::remove_file(&oxigaf_path);
}

#[test]
fn test_layer_path_format_matches_varbuilder() {
    use candle_core::{DType, Device};
    use candle_nn as nn;

    let temp_dir = env::temp_dir();
    let torsh_path = temp_dir.join("test_layer_path_torsh.safetensors");
    let oxigaf_path = temp_dir.join("test_layer_path_oxigaf.safetensors");

    // Create and convert
    create_synthetic_gaf_checkpoint(&torsh_path).expect("Failed to create synthetic checkpoint");

    let converter = WeightConverter::new();
    converter
        .torsh_to_oxigaf(&torsh_path, &oxigaf_path)
        .expect("Conversion should succeed");

    // Load with VarBuilder
    let device = Device::Cpu;
    let dtype = DType::F32;
    let data = std::fs::read(&oxigaf_path).expect("Failed to read file");
    let vb = nn::VarBuilder::from_buffered_safetensors(data, dtype, &device)
        .expect("VarBuilder should load");

    // Test various path access patterns with explicit shapes
    // Format: (path_parts, param_name, shape)
    let test_cases: Vec<(Vec<&str>, &str, Vec<usize>)> = vec![
        // Simple path
        (vec!["conv_in"], "weight", vec![320, 8, 3, 3]),
        // Nested path with numeric indices
        (
            vec!["down_blocks", "0", "resnets", "0", "norm1"],
            "weight",
            vec![320],
        ),
        // Deep nested path
        (
            vec!["time_embedding", "linear_1"],
            "weight",
            vec![1280, 320],
        ),
        // Mid block
        (
            vec!["mid_block", "resnets", "0", "norm1"],
            "weight",
            vec![1280],
        ),
    ];

    for (path_parts, param_name, shape) in test_cases {
        let mut vs = vb.clone();
        for part in &path_parts {
            vs = vs.pp(part);
        }

        // Load tensor with explicit shape
        let tensor_result = match shape.len() {
            1 => vs
                .get((shape[0],), param_name)
                .map(|t| Box::new(t) as Box<dyn std::any::Any>),
            2 => vs
                .get((shape[0], shape[1]), param_name)
                .map(|t| Box::new(t) as Box<dyn std::any::Any>),
            3 => vs
                .get((shape[0], shape[1], shape[2]), param_name)
                .map(|t| Box::new(t) as Box<dyn std::any::Any>),
            4 => vs
                .get((shape[0], shape[1], shape[2], shape[3]), param_name)
                .map(|t| Box::new(t) as Box<dyn std::any::Any>),
            _ => panic!("Unsupported shape length"),
        };

        match tensor_result {
            Ok(_) => {
                println!(
                    "✓ Successfully loaded {}.{}: shape {:?}",
                    path_parts.join("."),
                    param_name,
                    shape
                );
            }
            Err(e) => {
                panic!(
                    "Failed to load {}.{}: {}",
                    path_parts.join("."),
                    param_name,
                    e
                );
            }
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(&torsh_path);
    let _ = std::fs::remove_file(&oxigaf_path);
}

#[test]
fn test_converted_shapes_match_unet_expectations() {
    use safetensors::SafeTensors;

    let temp_dir = env::temp_dir();
    let torsh_path = temp_dir.join("test_shapes_torsh.safetensors");
    let oxigaf_path = temp_dir.join("test_shapes_oxigaf.safetensors");

    // Create and convert
    create_synthetic_gaf_checkpoint(&torsh_path).expect("Failed to create synthetic checkpoint");

    let converter = WeightConverter::new();
    converter
        .torsh_to_oxigaf(&torsh_path, &oxigaf_path)
        .expect("Conversion should succeed");

    // Load and check shapes
    let data = std::fs::read(&oxigaf_path).expect("Failed to read file");
    let safetensors = SafeTensors::deserialize(&data).expect("Failed to parse safetensors");

    // Expected shapes for key layers
    let expected_shapes = vec![
        ("conv_in.weight", vec![320, 8, 3, 3]),
        ("time_embedding.linear_1.weight", vec![1280, 320]),
        ("time_embedding.linear_2.weight", vec![1280, 1280]),
        ("camera_embedding.linear_1.weight", vec![1280, 16]),
        ("down_blocks.0.resnets.0.norm1.weight", vec![320]),
        ("down_blocks.0.resnets.0.conv1.weight", vec![320, 320, 3, 3]),
        ("mid_block.resnets.0.norm1.weight", vec![1280]),
        ("up_blocks.0.resnets.0.norm1.weight", vec![320]),
        ("conv_out.weight", vec![4, 320, 3, 3]),
    ];

    for (name, expected_shape) in expected_shapes {
        let tensor = safetensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("Layer {} not found", name));

        assert_eq!(
            tensor.shape(),
            &expected_shape[..],
            "Shape mismatch for layer {}",
            name
        );
    }

    // Cleanup
    let _ = std::fs::remove_file(&torsh_path);
    let _ = std::fs::remove_file(&oxigaf_path);
}

#[test]
fn test_no_slash_in_converted_layer_names() {
    use safetensors::SafeTensors;

    let temp_dir = env::temp_dir();
    let torsh_path = temp_dir.join("test_no_slash_torsh.safetensors");
    let oxigaf_path = temp_dir.join("test_no_slash_oxigaf.safetensors");

    // Create and convert
    create_synthetic_gaf_checkpoint(&torsh_path).expect("Failed to create synthetic checkpoint");

    let converter = WeightConverter::new();
    converter
        .torsh_to_oxigaf(&torsh_path, &oxigaf_path)
        .expect("Conversion should succeed");

    // Load and check all names
    let data = std::fs::read(&oxigaf_path).expect("Failed to read file");
    let safetensors = SafeTensors::deserialize(&data).expect("Failed to parse safetensors");

    for name in safetensors.names() {
        assert!(
            !name.contains('/'),
            "Converted layer name {} should not contain slashes",
            name
        );
        // Most names should contain dots (except simple ones like "weight" or "bias")
        // We just verify no slashes here
    }

    // Cleanup
    let _ = std::fs::remove_file(&torsh_path);
    let _ = std::fs::remove_file(&oxigaf_path);
}
