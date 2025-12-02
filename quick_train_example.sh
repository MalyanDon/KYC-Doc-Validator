#!/bin/bash
# Quick example of training positions

echo "📚 Training Position Detector - Example"
echo "========================================"
echo ""
echo "This script shows how to train positions from real documents"
echo ""
echo "Method 1: Auto-learn from images"
echo "  python train_positions.py --method images --input_dir data/train/aadhaar/ --doc_type aadhaar"
echo ""
echo "Method 2: Learn from annotations"
echo "  1. Annotate: python src/annotation_helper.py image.jpg"
echo "  2. Train: python train_positions.py --method annotations --input_dir annotations/ --doc_type aadhaar"
echo ""
echo "After training, learned positions are automatically used!"
