#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

OUTPUT_DIR="data/raw"
COMPETITION="aptos2019-blindness-detection"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Diabetic Retinopathy Dataset Downloader"
echo "============================================================"
echo ""

# Check Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo -e "${RED}✗ Error: Kaggle CLI not found${NC}"
    echo "Install with: pip install kaggle"
    exit 1
fi
echo -e "${GREEN}✓${NC} Kaggle CLI found"

# Check credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}✗ Error: Kaggle API credentials not found${NC}"
    echo ""
    echo "Setup instructions:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Place kaggle.json in ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
echo -e "${GREEN}✓${NC} Kaggle credentials found"
echo ""

echo -e "${BLUE}📥 Downloading APTOS 2019 Dataset${NC}"
echo "   Size: ~10 GB"
echo "   Images: 5,593 retinal images"
echo "   Classes: 5 (No DR, Mild, Moderate, Severe, Proliferative)"
echo "" 
        
echo "Downloading training images..."
kaggle competitions download -c "$COMPETITION" -p "$OUTPUT_DIR"
                
echo ""
echo -e "${BLUE}📦 Extracting...${NC}"
        
# Extract dataset
unzip -q "$OUTPUT_DIR/$COMPETITION.zip" -d $OUTPUT_DIR
rm "$OUTPUT_DIR/$COMPETITION.zip"

        
echo -e "${GREEN}✓ APTOS 2019 dataset ready!${NC}"

