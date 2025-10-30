#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Setup BAGEL data directory structure
# This script creates the external data directory structure for BAGEL

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "BAGEL Data Directory Setup"
echo "=========================================="
echo ""

# Check if BAGEL_DATA_DIR is set
if [ -z "$BAGEL_DATA_DIR" ]; then
    echo -e "${YELLOW}Warning: BAGEL_DATA_DIR not set in environment${NC}"
    echo "Please set it in your .env file or provide it as an argument"
    echo ""
    echo "Usage: $0 [data_directory]"
    echo "   or: BAGEL_DATA_DIR=/path/to/data $0"
    echo ""

    if [ -z "$1" ]; then
        echo -e "${RED}Error: No data directory specified${NC}"
        exit 1
    else
        BAGEL_DATA_DIR="$1"
        echo "Using provided directory: $BAGEL_DATA_DIR"
    fi
fi

echo "Creating data directory structure at:"
echo "  $BAGEL_DATA_DIR"
echo ""

# Create directory structure
echo "Creating directories..."
mkdir -p "$BAGEL_DATA_DIR"/{datasets/{t2i,editing/{seedxedit_multi,parquet_info},vlm/images},samples,outputs,test_images}

# Check if successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Directory structure created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create directory structure${NC}"
    exit 1
fi

echo ""
echo "Directory structure:"
tree -L 3 "$BAGEL_DATA_DIR" 2>/dev/null || find "$BAGEL_DATA_DIR" -type d | head -20

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory layout:"
echo "  $BAGEL_DATA_DIR/"
echo "  ├── datasets/      # Production datasets"
echo "  │   ├── t2i/       # Text-to-image data"
echo "  │   ├── editing/   # Image editing data"
echo "  │   └── vlm/       # Vision-language data"
echo "  ├── samples/       # Example/test data"
echo "  ├── outputs/       # Training outputs"
echo "  └── test_images/   # Test images for inference"
echo ""
echo "Next steps:"
echo "1. Copy your data to the appropriate directories"
echo "2. Update data/dataset_info.py if using custom dataset names"
echo "3. Source your .env file before training: source .env"
echo ""
