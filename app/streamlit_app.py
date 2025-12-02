"""
Streamlit UI for KYC Document Validator
Upload image/PDF → classify → extract text → detect fakes → output results
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import sys
import fitz  # PyMuPDF

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_ensemble_model, compile_model
from ocr_utils import extract_document_info, mock_uidai_validation, mock_it_validation
from fake_detector import comprehensive_fake_detection


# Page config
st.set_page_config(
    page_title="KYC Document Validator",
    page_icon="🆔",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model"""
    try:
        model = create_ensemble_model(input_shape=(150, 150, 3), num_classes=4)
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image: np.ndarray, target_size: tuple = (150, 150)) -> np.ndarray:
    """Preprocess image for model input"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image


def pdf_to_images(pdf_file) -> list:
    """Convert PDF to list of images"""
    images = []
    try:
        pdf_bytes = pdf_file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            images.append(img)
        
        pdf_doc.close()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    
    return images


def predict_document(image: np.ndarray, model) -> dict:
    """Predict document type and authenticity"""
    preprocessed = preprocess_image(image)
    input_batch = np.expand_dims(preprocessed, axis=0)
    
    predictions = model.predict(input_batch, verbose=0)
    
    class_probs = predictions[0][0]
    auth_score = float(predictions[1][0])
    
    class_names = ['Aadhaar', 'PAN', 'Fake', 'Other']
    predicted_class_idx = np.argmax(class_probs)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(class_probs[predicted_class_idx])
    
    return {
        'type': predicted_class,
        'confidence': confidence,
        'authenticity': auth_score,
        'all_probs': {name: float(prob) for name, prob in zip(class_names, class_probs)}
    }


def draw_highlights(image: np.ndarray, issues: list) -> np.ndarray:
    """Draw red boxes/highlights on image for detected issues"""
    highlighted = image.copy()
    h, w = image.shape[:2]
    
    # Draw border if issues found
    if issues:
        cv2.rectangle(highlighted, (0, 0), (w-1, h-1), (0, 0, 255), 3)
    
    # Add text annotations for issues
    y_offset = 30
    for i, issue in enumerate(issues[:5]):  # Limit to 5 issues
        cv2.putText(highlighted, issue, (10, y_offset + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return highlighted


def main():
    st.title("🆔 KYC Document Validator")
    st.markdown("Upload an image or PDF to classify and validate Indian ID documents (Aadhaar/PAN)")
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("Model Configuration")
        model_path = st.text_input(
            "Model Path",
            value="models/kyc_validator.h5",
            help="Path to trained model weights"
        )
        
        if st.button("Load Model"):
            if os.path.exists(model_path):
                with st.spinner("Loading model..."):
                    model = load_model(model_path)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model")
            else:
                st.warning(f"Model file not found at {model_path}")
                st.info("You can train a model using: python src/train.py")
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Model not loaded")
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Upload an image or PDF of Aadhaar or PAN card"
    )
    
    if uploaded_file is not None:
        if not st.session_state.model_loaded:
            st.warning("Please load the model first using the sidebar")
            return
        
        # Process uploaded file
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            st.info("Processing PDF file...")
            images = pdf_to_images(uploaded_file)
            if not images:
                st.error("Failed to extract images from PDF")
                return
            image = images[0]  # Process first page
            st.info(f"Processing first page of PDF ({len(images)} pages total)")
        else:
            image = np.array(Image.open(uploaded_file))
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Document")
            st.image(image, use_container_width=True)
        
        # Process document
        with st.spinner("Processing document..."):
            # Classification
            prediction = predict_document(image, st.session_state.model)
            
            # OCR
            ocr_info = extract_document_info(image)
            
            # Fake detection
            fake_detection = comprehensive_fake_detection(
                image,
                prediction['type'].lower(),
                ocr_info['raw_text']
            )
            
            # API validation (mock)
            api_validation = {}
            if ocr_info['aadhaar_number']:
                api_validation['aadhaar'] = mock_uidai_validation(ocr_info['aadhaar_number'])
            if ocr_info['pan_number']:
                api_validation['pan'] = mock_it_validation(ocr_info['pan_number'])
        
        # Create highlighted image
        all_issues = fake_detection['issues'].copy()
        if fake_detection['is_fake']:
            all_issues.append('FAKE_DOCUMENT')
        
        highlighted_image = draw_highlights(image, all_issues)
        
        with col2:
            st.subheader("Analysis Results")
            st.image(highlighted_image, use_container_width=True)
        
        # Results display
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        result_cols = st.columns(4)
        with result_cols[0]:
            st.metric("Document Type", prediction['type'])
        with result_cols[1]:
            st.metric("Type Confidence", f"{prediction['confidence']:.2%}")
        with result_cols[2]:
            st.metric("Authenticity Score", f"{prediction['authenticity']:.2%}")
        with result_cols[3]:
            auth_status = "✅ Real" if not fake_detection['is_fake'] else "❌ Fake"
            st.metric("Status", auth_status)
        
        # Detailed probabilities
        with st.expander("View All Class Probabilities"):
            for class_name, prob in prediction['all_probs'].items():
                st.progress(prob, text=f"{class_name}: {prob:.2%}")
        
        # OCR Results
        st.markdown("---")
        st.subheader("📝 Extracted Information")
        
        ocr_cols = st.columns(2)
        with ocr_cols[0]:
            if ocr_info['aadhaar_number']:
                st.success(f"**Aadhaar Number:** {ocr_info['aadhaar_number']}")
                if 'aadhaar' in api_validation:
                    st.info(f"UIDAI Validation: {api_validation['aadhaar']['message']}")
            else:
                st.info("No Aadhaar number detected")
        
        with ocr_cols[1]:
            if ocr_info['pan_number']:
                st.success(f"**PAN Number:** {ocr_info['pan_number']}")
                if 'pan' in api_validation:
                    st.info(f"IT Validation: {api_validation['pan']['message']}")
            else:
                st.info("No PAN number detected")
        
        # Fake Detection Results
        st.markdown("---")
        st.subheader("🔍 Fake Detection Analysis")
        
        fake_cols = st.columns(2)
        with fake_cols[0]:
            st.metric("Overall Authenticity", f"{fake_detection['authenticity_score']:.2%}")
            if fake_detection['is_fake']:
                st.error("⚠️ Document flagged as potentially fake!")
            else:
                st.success("✅ Document appears authentic")
        
        with fake_cols[1]:
            if fake_detection['issues']:
                st.warning("**Detected Issues:**")
                for issue in fake_detection['issues']:
                    st.write(f"- {issue.replace('_', ' ').title()}")
            else:
                st.success("No issues detected")
        
        # JSON Output
        st.markdown("---")
        st.subheader("📄 JSON Output")
        
        output_json = {
            "type": prediction['type'],
            "type_confidence": float(prediction['confidence']),
            "authenticity": float(fake_detection['authenticity_score']),
            "is_fake": fake_detection['is_fake'],
            "issues": fake_detection['issues'],
            "extracted_data": {
                "aadhaar_number": ocr_info['aadhaar_number'],
                "pan_number": ocr_info['pan_number'],
                "text_length": ocr_info['text_length']
            },
            "api_validation": api_validation
        }
        
        st.json(output_json)
        
        # Download JSON
        json_str = json.dumps(output_json, indent=2)
        st.download_button(
            label="Download JSON Results",
            data=json_str,
            file_name="kyc_validation_results.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()

