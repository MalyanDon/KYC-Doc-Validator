"""
Enhanced Streamlit Dashboard for KYC Document Validator
Upload image → Classify → Extract text → Position-based fake detection → Complete analysis
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models_enhanced import create_enhanced_ensemble_model, compile_enhanced_model
from ocr_utils import extract_document_info, mock_uidai_validation, mock_it_validation
from fake_detector import comprehensive_fake_detection
try:
    from position_based_fake_detector import detect_fake_using_positions
    from document_boundary_detector import draw_boundaries
except ImportError:
    # Fallback if position detector not available
    def detect_fake_using_positions(*args, **kwargs):
        return {'is_fake': False, 'confidence': 1.0, 'issues': [], 'position_details': {}}
    def draw_boundaries(image, boundaries):
        return image
try:
    from explicit_feature_validator import validate_with_explicit_features
except ImportError:
    # Fallback if explicit validator not available
    def validate_with_explicit_features(*args, **kwargs):
        return {'recommended_type': None, 'confidence_adjustment': 0.0}

# Page config
st.set_page_config(
    page_title="KYC Document Validator - Enhanced",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .status-real {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #c3e6cb;
    }
    .status-fake {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #f5c6cb;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'enhanced_model' not in st.session_state:
    st.session_state.enhanced_model = None


@st.cache_resource
def load_enhanced_model(model_path: str):
    """Load the enhanced trained model with position prediction"""
    try:
        model = create_enhanced_ensemble_model(
            input_shape=(150, 150, 3),
            num_classes=4,
            predict_positions=True
        )
        model = compile_enhanced_model(model, learning_rate=0.001, predict_positions=True)
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
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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


def predict_document_enhanced(image: np.ndarray, model) -> dict:
    """Predict document type, authenticity, and positions using enhanced model"""
    preprocessed = preprocess_image(image)
    input_batch = np.expand_dims(preprocessed, axis=0)
    
    predictions = model.predict(input_batch, verbose=0)
    
    # Enhanced model has 3 outputs: classification, authenticity, positions
    class_probs = predictions[0][0]
    auth_score = float(predictions[1][0][0])
    positions = predictions[2][0]  # Shape: (16,)
    
    class_names = ['Aadhaar', 'PAN', 'Fake', 'Other']
    all_probs_dict = {name: float(prob) for name, prob in zip(class_names, class_probs)}
    
    # Find highest probability
    predicted_class_idx = np.argmax(class_probs)
    max_confidence = float(class_probs[predicted_class_idx])
    
    # Check if probabilities are too close together (uncertain prediction)
    sorted_probs = sorted(class_probs, reverse=True)
    prob_diff = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    
    # Confidence threshold: If max confidence is too low OR probabilities are too close, classify as "Other"
    CONFIDENCE_THRESHOLD = 0.5  # 50% minimum confidence
    UNCERTAINTY_THRESHOLD = 0.15  # If top 2 probabilities differ by <15%, it's uncertain
    
    # Determine classification
    if max_confidence < CONFIDENCE_THRESHOLD:
        # Low confidence - classify as "Other"
        predicted_class = 'Other'
        confidence = max_confidence
        is_uncertain = True
        uncertainty_reason = f"Low confidence ({max_confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%} threshold)"
    elif prob_diff < UNCERTAINTY_THRESHOLD and max_confidence < 0.7:
        # Probabilities too close together - uncertain
        predicted_class = 'Other'
        confidence = max_confidence
        is_uncertain = True
        uncertainty_reason = f"Uncertain prediction (top probabilities differ by only {prob_diff:.1%})"
    else:
        # Confident prediction
        predicted_class = class_names[predicted_class_idx]
        confidence = max_confidence
        is_uncertain = False
        uncertainty_reason = None
    
    return {
        'type': predicted_class,
        'confidence': confidence,
        'authenticity': auth_score,
        'positions': positions,
        'all_probs': all_probs_dict,
        'is_uncertain': is_uncertain,
        'uncertainty_reason': uncertainty_reason,
        'original_prediction': class_names[predicted_class_idx],  # What argmax would have said
        'original_confidence': max_confidence
    }


def draw_position_overlay(image: np.ndarray, positions: dict, doc_type: str) -> np.ndarray:
    """Draw position overlays on image"""
    overlay = image.copy()
    h, w = image.shape[:2]
    
    colors = {
        'photo': (0, 255, 0),      # Green
        'name': (255, 0, 0),       # Blue
        'dob': (0, 0, 255),        # Red
        'document_number': (255, 255, 0)  # Cyan
    }
    
    labels = {
        'photo': 'Photo',
        'name': 'Name',
        'dob': 'DOB',
        'document_number': 'Number'
    }
    
    for element_name, pos_array in positions.items():
        if element_name in colors and len(pos_array) == 4:
            x_min, y_min, x_max, y_max = pos_array
            
            # Convert normalized to pixel coordinates
            x1 = int(x_min * w)
            y1 = int(y_min * h)
            x2 = int(x_max * w)
            y2 = int(y_max * h)
            
            # Draw rectangle
            color = colors[element_name]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            cv2.putText(overlay, labels[element_name], (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return overlay


def main():
    st.markdown('<div class="main-header">🆔 KYC Document Validator - Enhanced Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Upload an image or PDF to classify and validate Indian ID documents (Aadhaar/PAN) with position-based fake detection**")
    
    # Info box explaining the logic
    st.markdown("""
    <div class="info-box">
        <strong>📋 How It Works:</strong><br>
        1. <strong>Classify</strong> document type (PAN/Aadhaar/Fake/Other)<br>
        2. If PAN/Aadhaar: Use <strong>document-specific positions</strong> (PAN → PAN positions, Aadhaar → Aadhaar positions)<br>
        3. Compare predicted positions against <strong>learned positions from real documents</strong><br>
        4. Combine with other detection methods (color, borders, photo tampering, etc.)<br>
        5. Make final decision: <strong>Model authenticity ≥70% = AUTHENTIC</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        model_type = st.radio(
            "Select Model Type",
            ["Enhanced Model (with Position Prediction)", "Standard Model"],
            help="Enhanced model includes position prediction for better fake detection"
        )
        
        if model_type == "Enhanced Model (with Position Prediction)":
            model_path = st.text_input(
                "Enhanced Model Path",
                value="models/kyc_validator_enhanced.h5",
                help="Path to enhanced model weights"
            )
        else:
            model_path = st.text_input(
                "Model Path",
                value="models/kyc_validator.h5",
                help="Path to standard model weights"
            )
        
        if st.button("🔄 Load Model", type="primary"):
            if os.path.exists(model_path):
                with st.spinner("Loading model..."):
                    if model_type == "Enhanced Model (with Position Prediction)":
                        model = load_enhanced_model(model_path)
                        if model is not None:
                            st.session_state.enhanced_model = model
                            st.session_state.model_loaded = True
                            st.success("✅ Enhanced model loaded successfully!")
                        else:
                            st.error("❌ Failed to load model")
                    else:
                        st.info("Standard model loading not implemented in this enhanced version")
            else:
                st.warning(f"⚠️ Model file not found at {model_path}")
                st.info("Make sure you've trained the enhanced model first")
        
        st.markdown("---")
        
        if st.session_state.model_loaded:
            st.success("✅ **Model Ready**")
            st.info("Enhanced model with position prediction loaded")
        else:
            st.warning("⚠️ **Model not loaded**")
            st.info("Click 'Load Model' button above")
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("""
        - ✅ Document Classification
        - ✅ Authenticity Detection
        - ✅ Position-Based Fake Detection
        - ✅ OCR Text Extraction
        - ✅ Multi-method Analysis
        """)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "📤 Upload Document",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Upload an image or PDF of Aadhaar or PAN card"
    )
    
    if uploaded_file is not None:
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the enhanced model first using the sidebar")
            return
        
        # Process uploaded file
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            st.info("📄 Processing PDF file...")
            images = pdf_to_images(uploaded_file)
            if not images:
                st.error("❌ Failed to extract images from PDF")
                return
            image = images[0]  # Process first page
            st.info(f"📄 Processing first page of PDF ({len(images)} pages total)")
        else:
            image = np.array(Image.open(uploaded_file))
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Original Document")
            st.image(image, width='stretch', channels="RGB")
        
        # Process document
        with st.spinner("🔍 Processing document (this may take a moment)..."):
            # Enhanced prediction
            prediction = predict_document_enhanced(image, st.session_state.enhanced_model)
            
            # OCR
            ocr_info = extract_document_info(image)
            
            # Explicit feature validation (NEW!)
            explicit_validation = validate_with_explicit_features(
                image,
                prediction['type'],
                ocr_info
            )
            
            # Store original model prediction before any overrides
            # Get the actual highest probability class (what model really predicted)
            sorted_probs_model = sorted(prediction['all_probs'].items(), key=lambda x: x[1], reverse=True)
            model_predicted_class = sorted_probs_model[0][0] if sorted_probs_model else prediction['type']
            model_predicted_conf = sorted_probs_model[0][1] if sorted_probs_model else prediction['confidence']
            
            prediction['model_predicted_type'] = model_predicted_class
            prediction['model_predicted_confidence'] = model_predicted_conf
            
            # Adjust classification based on explicit features
            # IMPORTANT: Only override if explicit features STRONGLY disagree AND model confidence is not very high
            if explicit_validation.get('recommended_type') and explicit_validation['recommended_type'] != model_predicted_class:
                # Check if we should override
                explicit_score = max(
                    explicit_validation.get('pan_validation', {}).get('overall_pan_score', 0),
                    explicit_validation.get('aadhaar_validation', {}).get('overall_aadhaar_score', 0)
                )
                
                # Override logic:
                # IMPORTANT: Explicit features (content) matter more than visual patterns
                # If model predicts PAN/Aadhaar but NO content features found → It's NOT that document type
                # 
                # Override rules:
                # - If explicit score < 30%: Override (no content features = not that document type)
                # - If explicit score ≥ 50%: Trust model (content supports prediction)
                # - If explicit score 30-50%: Use model confidence to decide
                should_override = False
                
                if explicit_score < 0.3:
                    # Very low explicit score - no content features found
                    # Even if model is confident, if there's no PAN/Aadhaar content → It's "Other"
                    should_override = True
                elif explicit_score >= 0.5:
                    # Explicit features support model - trust both
                    should_override = False
                else:
                    # Medium explicit score (30-50%) - use model confidence
                    if model_predicted_conf >= 0.8:
                        # High model confidence + some explicit features → Trust model
                        should_override = False
                    else:
                        # Lower model confidence + medium explicit features → Override
                        should_override = True
                
                if should_override:
                    # Explicit features disagree with model - use explicit recommendation
                    prediction['type'] = explicit_validation['recommended_type']
                    prediction['explicit_override'] = True
                    prediction['original_prediction'] = model_predicted_class
                    prediction['confidence'] = max(0.0, model_predicted_conf + explicit_validation.get('confidence_adjustment', 0))
                else:
                    # Trust the model - keep model's prediction
                    prediction['type'] = model_predicted_class
                    prediction['explicit_override'] = False
                    prediction['confidence'] = model_predicted_conf
            else:
                # No override needed - use model's prediction
                prediction['type'] = model_predicted_class
                prediction['explicit_override'] = False
                prediction['confidence'] = model_predicted_conf
            
            # Position-based fake detection
            # IMPORTANT: Respect the model's classification
            # Only use PAN/Aadhaar positions if model actually classified it as PAN/Aadhaar with confidence
            predicted_type = prediction['type'].lower()
            
            # If uncertain or low confidence, don't use position detection
            if prediction.get('is_uncertain', False) or prediction['confidence'] < 0.5:
                predicted_type = 'other'  # Force to other if uncertain
            
            doc_type_for_detection = predicted_type  # Keep classification
            
            # Only run position detection and authenticity checks for Aadhaar or PAN (NOT for "Other")
            if predicted_type in ['aadhaar', 'pan']:
                # Run position detection
                position_result = detect_fake_using_positions(
                    image,
                    predicted_type,
                    st.session_state.enhanced_model
                )
                # Run comprehensive fake detection
                fake_detection = comprehensive_fake_detection(
                    image,
                    predicted_type,
                    ocr_info['raw_text'],
                    use_layout_validation=True
                )
            else:
                # Skip ALL authenticity checks for "Other" documents
                position_result = None
                fake_detection = None
                # Don't show authenticity scores for "Other"
                prediction['authenticity'] = None
            
            # API validation (mock)
            api_validation = {}
            if ocr_info['aadhaar_number']:
                api_validation['aadhaar'] = mock_uidai_validation(ocr_info['aadhaar_number'])
            if ocr_info['pan_number']:
                api_validation['pan'] = mock_it_validation(ocr_info['pan_number'])
        
        # Parse positions for visualization
        positions_dict = {}
        if 'positions' in prediction:
            pos_array = prediction['positions']
            positions_dict = {
                'photo': pos_array[0:4],
                'name': pos_array[4:8],
                'dob': pos_array[8:12],
                'document_number': pos_array[12:16]
            }
        
        # Draw position overlay
        position_overlay = draw_position_overlay(image.copy(), positions_dict, doc_type_for_detection)
        
        with col2:
            st.subheader("🎯 Document Boundaries & Position Analysis")
            
            # Show boundary detection if available
            if is_pan_or_aadhaar and position_result and position_result.get('boundary_detection', {}).get('detected'):
                boundary_info = position_result['boundary_detection']
                boundary_overlay = draw_boundaries(image.copy(), {
                    'boundaries': boundary_info['boundaries'],
                    'corners': [(0,0), (0,0), (0,0), (0,0)],  # Simplified
                    'confidence': boundary_info['confidence'],
                    'method': boundary_info['method']
                })
                st.image(boundary_overlay, width='stretch', channels="RGB", caption="Detected Document Boundaries")
                st.info(f"✅ Boundaries detected using **{boundary_info['method']}** method (confidence: {boundary_info['confidence']:.1%})")
            
            st.image(position_overlay, width='stretch', channels="RGB")
            st.caption("Green: Photo | Blue: Name | Red: DOB | Yellow: Document Number")
            if predicted_type in ['aadhaar', 'pan']:
                st.info(f"Using **{predicted_type.upper()}** positions for validation")
            else:
                st.warning(f"⚠️ Document classified as **{prediction['type']}**. Position detection only available for Aadhaar and PAN cards.")
        
        # Overall Status Banner
        st.markdown("---")
        
        # Only show authenticity status for PAN/Aadhaar documents
        is_pan_or_aadhaar = predicted_type in ['aadhaar', 'pan']
        
        # Initialize variables for "Other" documents
        overall_confidence = None
        is_fake_overall = False
        model_authenticity = None
        fake_detection_score = None
        position_confidence = None
        
        if is_pan_or_aadhaar:
            # Extract authenticity scores
            model_authenticity = float(prediction['authenticity'])
            fake_detection_score = float(fake_detection.get('authenticity_score', model_authenticity))
            position_confidence = float(position_result.get('confidence', 1.0))
            
            # Calculate weighted overall confidence
            overall_confidence = (
                fake_detection_score * 0.4 +
                position_confidence * 0.4 +
                model_authenticity * 0.2
            )
            
            # More nuanced fake detection logic
            is_fake_overall = False
            
            # Major red flag: Very low model authenticity
            if model_authenticity < 0.4:
                is_fake_overall = True
            # Multiple red flags: Both fake detection and position detection flag it
            elif fake_detection.get('is_fake', False) and position_result.get('is_fake', False):
                is_fake_overall = True
            # Low overall confidence with multiple issues
            elif overall_confidence < 0.6 and (fake_detection.get('is_fake', False) or position_result.get('is_fake', False)):
                is_fake_overall = True
            # If authenticity is high (≥70%), trust it unless position detection strongly disagrees
            elif model_authenticity >= 0.7:
                # Only flag as fake if position detection is very confident it's fake
                if position_result.get('is_fake', False) and position_confidence < 0.3:
                    is_fake_overall = True
                else:
                    is_fake_overall = False
            # Medium authenticity (50-70%): Use overall confidence
            else:
                is_fake_overall = overall_confidence < 0.6
            
            # Display status with more context
            if is_fake_overall:
                st.markdown(
                    f'<div class="status-fake">'
                    f'<h2>❌ FAKE DOCUMENT DETECTED</h2>'
                    f'<p>This document has been flagged as potentially fake.</p>'
                    f'<p><strong>Overall Confidence:</strong> {overall_confidence:.1%} | '
                    f'<strong>Model Authenticity:</strong> {model_authenticity:.1%}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="status-real">'
                    f'<h2>✅ AUTHENTIC DOCUMENT</h2>'
                    f'<p>This document appears to be authentic based on our analysis.</p>'
                    f'<p><strong>Overall Confidence:</strong> {overall_confidence:.1%} | '
                    f'<strong>Model Authenticity:</strong> {model_authenticity:.1%}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        else:
            # For "Other" documents, just show classification
            st.markdown(
                f'<div class="status-real">'
                f'<h2>📄 DOCUMENT CLASSIFIED</h2>'
                f'<p>Document Type: <strong>{prediction["type"].upper()}</strong></p>'
                f'<p>This document is not a PAN or Aadhaar card. Authenticity checks are only available for PAN and Aadhaar documents.</p>'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        # Main Results Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔍 Fake Detection", 
            "📍 Position Analysis", 
            "📝 OCR Results", 
            "📄 Full Report"
        ])
        
        with tab1:
            st.subheader("📊 Classification & Overview")
            
            overview_cols = st.columns(4)
            with overview_cols[0]:
                doc_type_display = prediction['type']
                if prediction.get('explicit_override', False):
                    doc_type_display += " 🔄"
                elif prediction.get('is_uncertain', False):
                    doc_type_display += " ⚠️"
                st.metric("Document Type", doc_type_display, delta=None)
                # Show model prediction if different
                if prediction.get('explicit_override', False):
                    st.caption(f"Model predicted: {prediction.get('model_predicted_type', 'Unknown')} ({prediction.get('model_predicted_confidence', 0):.1%})")
            with overview_cols[1]:
                st.metric("Type Confidence", f"{prediction['confidence']:.1%}")
            with overview_cols[2]:
                if is_pan_or_aadhaar:
                    model_authenticity = float(prediction.get('authenticity', 0))
                    st.metric("Model Authenticity", f"{model_authenticity:.1%}")
                else:
                    st.metric("Authenticity Check", "N/A")
                    st.caption("Only for PAN/Aadhaar")
            with overview_cols[3]:
                if is_pan_or_aadhaar:
                    overall_confidence = (
                        float(fake_detection.get('authenticity_score', prediction.get('authenticity', 0))) * 0.4 +
                        float(position_result.get('confidence', 1.0)) * 0.4 +
                        float(prediction.get('authenticity', 0)) * 0.2
                    )
                    st.metric("Overall Confidence", f"{overall_confidence:.1%}")
                else:
                    st.metric("Overall Confidence", "N/A")
                    st.caption("Only for PAN/Aadhaar")
            
            # Show uncertainty warning
            if prediction.get('is_uncertain', False):
                st.warning(f"⚠️ **Uncertain Classification:** {prediction.get('uncertainty_reason', 'Low confidence')}")
                st.info(f"Original prediction would have been: **{prediction.get('original_prediction', 'Unknown')}** ({prediction.get('original_confidence', 0):.1%})")
                st.write("**Reason:** Model is not confident enough to classify as PAN/Aadhaar. Classified as 'Other' instead.")
            
            st.markdown("---")
            
            # Only show probability breakdown for PAN/Aadhaar documents
            # Hide confusing details for "Other" documents
            if is_pan_or_aadhaar:
                # Class probabilities with better visualization
                st.subheader("Classification Probabilities")
                
                # Sort by probability for better visualization
                sorted_probs = sorted(prediction['all_probs'].items(), key=lambda x: x[1], reverse=True)
                
                for class_name, prob in sorted_probs:
                    # Highlight the final classification
                    if class_name == prediction['type']:
                        st.progress(prob, text=f"**{class_name}: {prob:.2%}** ← Classified")
                    else:
                        st.progress(prob, text=f"{class_name}: {prob:.2%}")
                
                # Show probability differences
                if len(sorted_probs) >= 2:
                    top_prob = sorted_probs[0][1]
                    second_prob = sorted_probs[1][1]
                    diff = top_prob - second_prob
                    st.caption(f"Difference between top 2: {diff:.1%} (smaller = more uncertain)")
            else:
                # For "Other" documents, just show final classification clearly
                st.subheader("Classification Result")
                st.success(f"✅ **Document Type: {prediction['type'].upper()}**")
                st.info(f"Confidence: {prediction['confidence']:.1%}")
                st.write("This document is not a PAN or Aadhaar card. Detailed probability breakdown is hidden to avoid confusion.")
            
            # Logic explanation (only for PAN/Aadhaar)
            if is_pan_or_aadhaar:
                st.markdown("---")
                st.subheader("🔍 Decision Logic")
                model_authenticity = float(prediction.get('authenticity', 0))
                overall_confidence = (
                    float(fake_detection.get('authenticity_score', prediction.get('authenticity', 0))) * 0.4 +
                    float(position_result.get('confidence', 1.0)) * 0.4 +
                    float(prediction.get('authenticity', 0)) * 0.2
                )
                
                if model_authenticity >= 0.7:
                    st.success(f"✅ **Model authenticity is {model_authenticity:.1%} (≥70%)** → Document is AUTHENTIC")
                elif model_authenticity < 0.4:
                    st.error(f"❌ **Model authenticity is {model_authenticity:.1%} (<40%)** → Document is FAKE")
                else:
                    st.info(f"⚠️ **Model authenticity is {model_authenticity:.1%} (40-70%)** → Using combined analysis")
                    st.write(f"Overall confidence: {overall_confidence:.1%}")
                
                # Explicit feature validation results
                st.markdown("---")
                st.subheader("🔍 Explicit Feature Validation")
                
                if prediction.get('explicit_override'):
                    st.warning(f"⚠️ **Classification Overridden!**")
                    st.write(f"**Model predicted:** {prediction.get('original_prediction', 'Unknown')}")
                    st.write(f"**Explicit features suggest:** {prediction['type']}")
                    issue_msg = explicit_validation.get('explicit_features', {}).get('issue', 'Features don\'t match prediction')
                    st.write(f"**Reason:** {issue_msg}")
                else:
                    st.success("✅ **Explicit features support model prediction**")
                
                # Show PAN validation
                if 'pan_validation' in explicit_validation:
                    pan_val = explicit_validation['pan_validation']
                    with st.expander("PAN Card Features Check"):
                        st.write(f"**PAN Format Valid:** {pan_val.get('pan_format_valid', False)} ({pan_val.get('pan_format_confidence', 0):.0%})")
                        st.write(f"**PAN Keywords Found:** {pan_val.get('pan_keywords_found', False)} ({pan_val.get('pan_keywords_confidence', 0):.0%})")
                        st.write(f"**Extracted PAN Valid:** {pan_val.get('extracted_pan_valid', False)}")
                        st.write(f"**Overall PAN Score:** {pan_val.get('overall_pan_score', 0):.0%}")
                
                # Show Aadhaar validation
                if 'aadhaar_validation' in explicit_validation:
                    aadhaar_val = explicit_validation['aadhaar_validation']
                    with st.expander("Aadhaar Card Features Check"):
                        st.write(f"**Aadhaar Format Valid:** {aadhaar_val.get('aadhaar_format_valid', False)} ({aadhaar_val.get('aadhaar_format_confidence', 0):.0%})")
                        st.write(f"**Aadhaar Keywords Found:** {aadhaar_val.get('aadhaar_keywords_found', False)} ({aadhaar_val.get('aadhaar_keywords_confidence', 0):.0%})")
                        st.write(f"**Extracted Aadhaar Valid:** {aadhaar_val.get('extracted_aadhaar_valid', False)}")
                        st.write(f"**Overall Aadhaar Score:** {aadhaar_val.get('overall_aadhaar_score', 0):.0%}")
            else:
                # For "Other" documents, skip authenticity logic
                st.markdown("---")
                st.info("ℹ️ Authenticity checks are only available for PAN and Aadhaar documents.")
        
        with tab2:
            if is_pan_or_aadhaar:
                st.subheader("🔍 Comprehensive Fake Detection")
                
                fake_detection_score = float(fake_detection.get('authenticity_score', prediction.get('authenticity', 0)))
                model_authenticity = float(prediction.get('authenticity', 0))
                position_confidence = float(position_result.get('confidence', 1.0))
                overall_confidence = (
                    fake_detection_score * 0.4 +
                    position_confidence * 0.4 +
                    model_authenticity * 0.2
                )
                is_fake_overall = (
                    model_authenticity < 0.4 or
                    (fake_detection.get('is_fake', False) and position_result.get('is_fake', False)) or
                    (overall_confidence < 0.6 and (fake_detection.get('is_fake', False) or position_result.get('is_fake', False)))
                )
                
                fake_cols = st.columns(3)
                
                with fake_cols[0]:
                    st.metric("Overall Authenticity", f"{fake_detection_score:.1%}")
                    st.metric("Model Authenticity", f"{model_authenticity:.1%}")
                    if fake_detection.get('is_fake', False):
                        st.warning("⚠️ Some issues detected")
                        st.info(f"Note: Overall confidence is {overall_confidence:.1%}")
                    else:
                        st.success("✅ Appears authentic")
                
                with fake_cols[1]:
                    st.metric("Position-Based Detection", f"{position_confidence:.1%}")
                    if position_result.get('is_fake', False):
                        st.error("⚠️ Position anomalies detected!")
                    else:
                        st.success("✅ Positions valid")
                    st.caption(f"Using {predicted_type.upper()} positions")
                
                with fake_cols[2]:
                    st.metric("Final Decision", "FAKE" if is_fake_overall else "AUTHENTIC")
                    st.caption(f"Based on: Model ({model_authenticity:.1%}) + Other methods")
                
                st.markdown("---")
                
                # Detection methods breakdown
                st.subheader("Detection Methods Breakdown")
                
                if 'detailed_results' in fake_detection:
                    methods = fake_detection['detailed_results']
                    
                    method_cols = st.columns(2)
                    
                    with method_cols[0]:
                        st.write("**Color Analysis:**")
                        if 'color_analysis' in methods:
                            st.write(f"- Confidence: {methods['color_analysis'].get('confidence', 0):.1%}")
                            if methods['color_analysis'].get('issues'):
                                st.write(f"- Issues: {', '.join(methods['color_analysis']['issues'])}")
                        
                        st.write("**Border Detection:**")
                        if 'border_detection' in methods:
                            st.write(f"- Confidence: {methods['border_detection'].get('confidence', 0):.1%}")
                            if methods['border_detection'].get('issues'):
                                st.write(f"- Issues: {', '.join(methods['border_detection']['issues'])}")
                        
                        st.write("**Photo Tampering:**")
                        if 'photo_tampering' in methods:
                            st.write(f"- Confidence: {methods['photo_tampering'].get('confidence', 0):.1%}")
                            if methods['photo_tampering'].get('issues'):
                                st.write(f"- Issues: {', '.join(methods['photo_tampering']['issues'])}")
                    
                    with method_cols[1]:
                        st.write("**QR Code Validation:**")
                        if 'qr_validation' in methods:
                            st.write(f"- Confidence: {methods['qr_validation'].get('confidence', 0):.1%}")
                            if methods['qr_validation'].get('issues'):
                                st.write(f"- Issues: {', '.join(methods['qr_validation']['issues'])}")
                        
                        st.write("**Layout Analysis:**")
                        if 'layout_analysis' in methods:
                            st.write(f"- Confidence: {methods['layout_analysis'].get('confidence', 0):.1%}")
                            if methods['layout_analysis'].get('issues'):
                                st.write(f"- Issues: {', '.join(methods['layout_analysis']['issues'])}")
                
                # All issues
                st.markdown("---")
                st.subheader("All Detected Issues")
                all_issues = fake_detection.get('issues', [])
                if position_result and position_result.get('issues'):
                    all_issues.extend(position_result['issues'])
                
                if all_issues:
                    for issue in set(all_issues):
                        st.warning(f"⚠️ {issue.replace('_', ' ').title()}")
                else:
                    st.success("✅ No issues detected")
            else:
                st.subheader("🔍 Fake Detection")
                st.info("ℹ️ Authenticity checks are only available for PAN and Aadhaar documents.")
                st.write(f"This document is classified as **{prediction['type'].upper()}**, which is not a PAN or Aadhaar card.")
                st.write("Fake detection analysis requires PAN or Aadhaar card structure and features.")
        
        with tab3:
            if is_pan_or_aadhaar:
                st.subheader("📍 Position-Based Analysis")
                st.info(f"**Using {predicted_type.upper()} card positions** (learned from real documents)")
                
                if 'position_details' in position_result and position_result['position_details']:
                    st.write("**Element Position Validation:**")
                    
                    for element_name, details in position_result['position_details'].items():
                        with st.expander(f"🔍 {element_name.upper()} Position", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                status = "✅ VALID" if details['is_valid'] else "❌ SUSPICIOUS"
                                st.write(f"**Status:** {status}")
                                st.write(f"**Overlap Ratio:** {details['overlap_ratio']:.1%}")
                                st.write(f"**Max Z-Score:** {details['z_score_max']:.2f} σ")
                            
                            with col2:
                                st.write("**Predicted Position:**")
                                st.write(f"X: [{details['predicted'][0]:.3f}, {details['predicted'][2]:.3f}]")
                                st.write(f"Y: [{details['predicted'][1]:.3f}, {details['predicted'][3]:.3f}]")
                                
                                st.write("**Expected Position:**")
                                st.write(f"X: [{details['learned_mean'][0]:.3f}, {details['learned_mean'][2]:.3f}]")
                                st.write(f"Y: [{details['learned_mean'][1]:.3f}, {details['learned_mean'][3]:.3f}]")
                            
                            if not details['is_valid']:
                                st.error("⚠️ Position does not match expected layout!")
                                if details['overlap_ratio'] < 0.5:
                                    st.warning(f"Low overlap ({details['overlap_ratio']:.1%}) - element may be in wrong location")
                                if details['z_score_max'] > 2.0:
                                    st.warning(f"High deviation ({details['z_score_max']:.2f}σ) - unusual position detected")
                    
                    if 'summary' in position_result:
                        st.markdown("---")
                        st.subheader("Position Analysis Summary")
                        summary = position_result['summary']
                        st.write(f"**Elements Checked:** {summary['total_elements_checked']}")
                        st.write(f"**Valid Elements:** {summary['valid_elements']}")
                        st.write(f"**Suspicious Elements:** {summary['suspicious_elements']}")
                else:
                    st.info("Position analysis details not available")
            else:
                st.subheader("📍 Position-Based Analysis")
                st.info("ℹ️ Position analysis is only available for PAN and Aadhaar documents.")
                st.write(f"This document is classified as **{prediction['type'].upper()}**, which is not a PAN or Aadhaar card.")
        
        with tab4:
            st.subheader("📝 Extracted Information (OCR)")
            
            ocr_cols = st.columns(2)
            
            with ocr_cols[0]:
                st.write("**Aadhaar Information:**")
                if ocr_info['aadhaar_number']:
                    st.success(f"**Aadhaar Number:** {ocr_info['aadhaar_number']}")
                    if 'aadhaar' in api_validation:
                        st.info(f"UIDAI Validation: {api_validation['aadhaar']['message']}")
                else:
                    st.info("No Aadhaar number detected")
            
            with ocr_cols[1]:
                st.write("**PAN Information:**")
                if ocr_info['pan_number']:
                    st.success(f"**PAN Number:** {ocr_info['pan_number']}")
                    if 'pan' in api_validation:
                        st.info(f"IT Validation: {api_validation['pan']['message']}")
                else:
                    st.info("No PAN number detected")
            
            st.markdown("---")
            st.subheader("Raw OCR Text")
            st.text_area("Extracted Text", ocr_info.get('raw_text', ''), height=200, label_visibility="collapsed")
        
        with tab5:
            st.subheader("📄 Complete Analysis Report")
            
            # Helper function to convert numpy types to Python native types for JSON
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_to_native(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                return obj
            
            # Ensure overall_confidence is defined
            if overall_confidence is None:
                if is_pan_or_aadhaar:
                    # Calculate it now
                    overall_confidence = (
                        float(fake_detection.get('authenticity_score', prediction.get('authenticity', 0))) * 0.4 +
                        float(position_result.get('confidence', 1.0)) * 0.4 +
                        float(prediction.get('authenticity', 0)) * 0.2
                    ) if fake_detection and position_result else None
                else:
                    overall_confidence = None
            
            output_json = {
                "document_type": prediction['type'],
                "type_confidence": float(prediction['confidence']),
                "overall_authenticity": float(overall_confidence) if overall_confidence is not None else None,
                "is_fake": bool(is_fake_overall) if is_pan_or_aadhaar else None,
                "decision_logic": {
                    "model_authenticity": float(model_authenticity) if model_authenticity is not None else None,
                    "threshold_used": "≥70% = AUTHENTIC, <40% = FAKE, else combined" if is_pan_or_aadhaar else "N/A - Not PAN/Aadhaar",
                    "final_decision": ("AUTHENTIC" if not is_fake_overall else "FAKE") if is_pan_or_aadhaar else "N/A"
                },
                "classification": {
                    "predicted_type": prediction['type'],
                    "confidence": float(prediction['confidence']),
                    "is_uncertain": bool(prediction.get('is_uncertain', False)),
                    "uncertainty_reason": prediction.get('uncertainty_reason'),
                    "original_prediction": prediction.get('original_prediction'),
                    "original_confidence": float(prediction.get('original_confidence', prediction['confidence'])),
                    "all_probabilities": {k: float(v) for k, v in prediction['all_probs'].items()},
                    "model_authenticity": float(prediction['authenticity']) if prediction.get('authenticity') is not None else None
                },
                "fake_detection": {
                    "is_fake": bool(fake_detection.get('is_fake', False)) if fake_detection else None,
                    "authenticity_score": float(fake_detection_score) if fake_detection_score is not None else None,
                    "issues": list(fake_detection.get('issues', [])) if fake_detection else [],
                    "detailed_results": convert_to_native(fake_detection.get('detailed_results', {})) if fake_detection else {}
                },
                "position_analysis": {
                    "document_type_used": predicted_type if predicted_type in ['aadhaar', 'pan'] else None,
                    "original_classification": prediction['type'],
                    "is_fake": bool(position_result.get('is_fake', False)),
                    "confidence": float(position_confidence),
                    "issues": list(position_result.get('issues', [])),
                    "position_details": convert_to_native(position_result.get('position_details', {})),
                    "summary": convert_to_native(position_result.get('summary', {}))
                },
                "extracted_data": {
                    "aadhaar_number": ocr_info.get('aadhaar_number'),
                    "pan_number": ocr_info.get('pan_number'),
                    "text_length": int(ocr_info.get('text_length', 0))
                },
                "api_validation": api_validation
            }
            
            st.json(output_json)
            
            # Download JSON
            json_str = json.dumps(output_json, indent=2, default=str)
            st.download_button(
                label="📥 Download Complete Report (JSON)",
                data=json_str,
                file_name="kyc_validation_report.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
