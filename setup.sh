#!/bin/bash

# KYC Document Validator Setup Script

echo "🚀 Setting up KYC Document Validator..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data/{train,val,test}/{aadhaar,pan,fake,other}

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Add your training images to data/train/, data/val/, and data/test/"
echo "3. Train the model: python src/train.py"
echo "4. Run the Streamlit app: streamlit run app/streamlit_app.py"

