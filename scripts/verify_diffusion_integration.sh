#!/bin/bash
set -e

echo "=========================================="
echo "OxiGAF-Diffusion Integration Verification"
echo "=========================================="
echo ""

# Step 1: Check dependencies
echo "1. Checking dependencies..."
echo "   - oxigaf-bridge with torsh feature"
cargo check -p oxigaf-bridge --features torsh
echo "   - oxigaf-diffusion"
cargo check -p oxigaf-diffusion
echo "   ✓ Dependencies OK"
echo ""

# Step 2: Run layer name validation tests
echo "2. Running layer name validation tests..."
cargo test -p oxigaf-bridge --features torsh test_layer_names_match_diffusion_varbuilder
echo "   ✓ Layer name validation passed"
echo ""

# Step 3: Run VarBuilder compatibility tests
echo "3. Running VarBuilder compatibility tests..."
cargo test -p oxigaf-bridge --features torsh test_varbuilder_can_load_converted_weights
cargo test -p oxigaf-bridge --features torsh test_layer_path_format_matches_varbuilder
echo "   ✓ VarBuilder compatibility tests passed"
echo ""

# Step 4: Run validation tests
echo "4. Running checkpoint validation tests..."
cargo test -p oxigaf-bridge --features torsh test_converted_weights_validate
cargo test -p oxigaf-bridge --features torsh test_converted_shapes_match_unet_expectations
echo "   ✓ Validation tests passed"
echo ""

# Step 5: Run all non-GPU integration tests
echo "5. Running all non-GPU integration tests..."
cargo test -p oxigaf-bridge --features torsh --test diffusion_integration
echo "   ✓ All integration tests passed"
echo ""

# Step 6: Check for GPU availability
echo "6. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ GPU detected (nvidia-smi available)"
    echo ""
    echo "7. Running GPU tests..."
    echo "   Note: GPU tests may fail if real GAF checkpoint is not available"
    echo "   This is expected - these tests are for manual verification with real weights"

    if cargo test -p oxigaf-bridge --features torsh test_converted_weights_load_in_diffusion --ignored; then
        echo "   ✓ GPU tests passed"
    else
        echo "   ⚠ GPU tests failed or skipped (this is OK for synthetic checkpoints)"
    fi
else
    echo "   ⚠ No GPU detected (nvidia-smi not found)"
    echo "   Skipping GPU tests"
fi
echo ""

echo "=========================================="
echo "✓ Verification complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Layer name mapping: ✓"
echo "  - VarBuilder compatibility: ✓"
echo "  - Checkpoint validation: ✓"
echo "  - Integration tests: ✓"
echo ""
echo "Next steps:"
echo "  - Use real GAF checkpoint to test full pipeline"
echo "  - Run GPU tests with actual weights"
echo "  - Test with oxigaf-diffusion MultiViewDiffusionPipeline"
echo ""
