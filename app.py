import streamlit as st
import json
import pandas as pd
from io import BytesIO
import zipfile
from PIL import Image

# Import functions from our new modules
from modules.ocr_utils import perform_ocr_on_image, reconstruct_text_from_ocr, get_paddle_ocr_model, preprocess_image
from modules.llm_utils import infer_document_schema, extract_data_with_inferred_schema, post_process_extracted_data, get_llm_model, self_correct_extracted_data

# --- Robust Global Model Initialization ---
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "paddle_ocr_model" not in st.session_state:
    st.session_state.paddle_ocr_model = None

# Initialize application stage
if "app_stage" not in st.session_state:
    st.session_state.app_stage = 'initial'
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "all_extracted_documents" not in st.session_state:
    st.session_state.all_extracted_documents = [] # To store results for all documents

# Now call the getter functions to initialize the actual models if they are None
get_llm_model()
get_paddle_ocr_model()

# --- Streamlit Application Layout ---
st.set_page_config(layout="wide", page_title="Universal Document Scanner (LLM Powered)")

st.title("📄 Universal Document Scanner (LLM Powered)")
st.write("Upload a single image or a `.zip` file containing multiple document images. Our AI will automatically infer the structure and extract key data.")

st.info("💡 **Feature Highlight**: This application leverages advanced **PaddleOCR** with comprehensive image pre-processing (grayscale, blur, binarization, and **deskewing**) for maximal text extraction accuracy on complex or scanned documents. "
          "A **Google Gemini/Gemma 3B Large Language Model (LLM)** then dynamically infers the document's structure and extracts key data, making it highly adaptable to various document layouts and types (e.g., invoices, forms, reports).")


uploaded_file = st.file_uploader("Upload Document(s) (JPG, PNG, or ZIP)", type=["jpg", "jpeg", "png", "zip"])

# If a new file is uploaded, reset the stage and store the file info
if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.app_stage = 'initial' # Reset stage on new upload
    st.session_state.inferred_schema_string = None
    st.session_state.document_text_for_current_document = None
    st.session_state.extracted_single_doc_data = None # Clear previous single doc data
    st.session_state.all_extracted_documents = [] # Clear previous bulk data
    # Rerun to ensure the file is processed from scratch
    st.rerun()

# --- New function to process a single document ---
def process_and_extract_document(file_name, image_bytes, progress_bar_container=None):
    """
    Handles the OCR, schema inference, and data extraction for a single document.
    Returns the extracted data or None if an error occurs.
    """
    st.subheader(f"Processing: {file_name}")

    if st.session_state.llm_model is None or st.session_state.paddle_ocr_model is None:
        st.error(f"Cannot process {file_name}: LLM or PaddleOCR model not initialized. Check API keys/dependencies.")
        return None

    current_progress = 0
    if progress_bar_container:
        progress_text = progress_bar_container.empty()
        progress_bar = progress_bar_container.progress(0)
    else:
        # Fallback if no container provided, e.g., for initial schema inference button
        progress_text = st.empty()
        progress_bar = st.progress(0)

    try:
        progress_text.text(f"Performing PaddleOCR on {file_name}...")
        progress_bar.progress(10)

        ocr_words = perform_ocr_on_image(image_bytes)

        if not ocr_words:
            st.warning(f"Could not extract any text from {file_name} via PaddleOCR. Skipping this document.")
            progress_text.empty()
            progress_bar.empty()
            return None
        
        document_text = reconstruct_text_from_ocr(ocr_words)
        
        progress_text.text(f"OCR complete for {file_name}. Inferring schema...")
        progress_bar.progress(40)

        # For batch processing, we infer schema for each document.
        # If schema inference is consistent across documents, you might infer once
        # and reuse, but for true "universal" handling, inferring per doc is better.
        inferred_schema = infer_document_schema(document_text)

        if not inferred_schema:
            st.error(f"Failed to infer schema for {file_name}. Skipping extraction for this document.")
            progress_text.empty()
            progress_bar.empty()
            return None

        with st.expander(f"View Raw Text and Inferred Schema for {file_name}"):
            st.write("--- Raw Text ---")
            st.code(document_text[:2000] + "..." if len(document_text) > 2000 else document_text)
            st.write("--- Inferred Schema ---")
            st.json(inferred_schema) # Display inferred schema for each document

        progress_text.text(f"Extracting raw data for {file_name}...")
        progress_bar.progress(60)
        raw_extracted_data = extract_data_with_inferred_schema(document_text, inferred_schema)

        progress_text.text(f"Applying self-correction to extracted data for {file_name}...")
        progress_bar.progress(80)
        corrected_extracted_data = self_correct_extracted_data(raw_extracted_data, document_text, inferred_schema)

        progress_text.text(f"Post-processing extracted data for {file_name}...")
        progress_bar.progress(90)
        final_extracted_data = post_process_extracted_data(corrected_extracted_data, inferred_schema)
        
        progress_text.text(f"Data extraction complete for {file_name}!")
        progress_bar.progress(100)
        progress_text.empty()
        progress_bar.empty()

        return final_extracted_data

    except Exception as e:
        st.error(f"An error occurred while processing {file_name}: {e}")
        if progress_bar_container:
            progress_bar_container.empty()
        else:
            progress_text.empty()
            progress_bar.empty()
        return None

# --- Main App Logic ---
if st.session_state.uploaded_file_bytes:
    # Use stored file info
    file_name = st.session_state.uploaded_file_name
    image_bytes = st.session_state.uploaded_file_bytes

    processed_images_info = []
    if file_name.lower().endswith(".zip"):
        with zipfile.ZipFile(BytesIO(image_bytes), 'r') as zip_ref:
            image_files_in_zip = [
                f for f in zip_ref.namelist()
                if not f.startswith('__MACOSX/') and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not image_files_in_zip:
                st.warning("No supported image files found in the ZIP archive.")
            else:
                st.info(f"Found {len(image_files_in_zip)} image(s) in the ZIP. Preparing for processing...")
                for zf_name in image_files_in_zip:
                    with zip_ref.open(zf_name) as image_file:
                        processed_images_info.append({"name": zf_name, "bytes": image_file.read()})
                st.success(f"All {len(processed_images_info)} images from ZIP ready.")
    else:
        processed_images_info.append({"name": file_name, "bytes": image_bytes})
    
    # --- Display Preview for the FIRST document only (for UX) ---
    st.subheader(f"Preview of first document: {processed_images_info[0]['name']}")
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_images_info[0]["bytes"], caption=f"Original {processed_images_info[0]['name']}", use_container_width=True) 

    with col2:
        processed_cv2_image = preprocess_image(processed_images_info[0]["bytes"])
        if processed_cv2_image is not None:
            pil_img = Image.fromarray(processed_cv2_image)
            st.image(pil_img, caption=f"Pre-processed {processed_images_info[0]['name']} (Enhanced for OCR)", use_container_width=True)
        else:
            st.write("Could not display pre-processed image.")

    # --- Bulk Processing Button ---
    if st.button("Start Batch Processing (OCR & Extract All)", key="start_batch_processing"):
        st.session_state.app_stage = 'processing_batch'
        st.session_state.all_extracted_documents = [] # Reset for new batch

        total_files = len(processed_images_info)
        overall_progress_text = st.empty()
        overall_progress_bar = st.progress(0, text="Overall Batch Progress")
        
        for i, doc_info in enumerate(processed_images_info):
            doc_name = doc_info["name"]
            doc_bytes = doc_info["bytes"]

            overall_progress_text.text(f"Processing document {i+1}/{total_files}: {doc_name}")
            
            # Create a dedicated container for progress of individual document within the overall progress
            with st.container():
                extracted_data = process_and_extract_document(doc_name, doc_bytes, progress_bar_container=st.empty())
            
            if extracted_data:
                # Add original filename to the extracted data
                extracted_data['originalFileName'] = doc_name 
                st.session_state.all_extracted_documents.append(extracted_data)
            
            overall_progress_bar.progress((i + 1) / total_files)
        
        overall_progress_text.empty()
        overall_progress_bar.empty()
        st.success(f"Batch processing complete! Processed {len(st.session_state.all_extracted_documents)}/{total_files} documents.")
        st.session_state.app_stage = 'batch_processed'
        st.rerun()

    # --- Display Results After Batch Processing ---
    if st.session_state.app_stage == 'batch_processed' and st.session_state.all_extracted_documents:
        st.subheader("Extracted Data from All Documents:")
        all_flattened_data = []

        for i, doc_data in enumerate(st.session_state.all_extracted_documents):
            # Flatten the main document fields
            flattened_doc = {k: v for k, v in doc_data.items() if k != 'items'}
            
            # Extract line items (renamed from 'lineItems' to 'items' as per LLM output)
            items = doc_data.get('items', []) # Corrected key
            
            if items:
                for item_idx, item in enumerate(items):
                    row = {**flattened_doc, **item}
                    row['itemNumber'] = item_idx + 1 # Add an item number for clarity
                    all_flattened_data.append(row)
            else:
                # If no line items, still include main doc data
                all_flattened_data.append(flattened_doc)
        
        if all_flattened_data:
            df = pd.DataFrame(all_flattened_data)
            
            # Build the columns list conditionally to avoid KeyError
            cols_to_display = []
            if 'originalFileName' in df.columns:
                cols_to_display.append('originalFileName')
            if 'itemNumber' in df.columns:
                cols_to_display.append('itemNumber')
            
            # Add all other columns that are not already in our list
            cols_to_display += [col for col in df.columns if col not in cols_to_display]

            df = df[cols_to_display]
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Extracted Data as CSV",
                data=csv_data,
                file_name="extracted_document_data.csv",
                mime="text/csv",
            )
        else:
            st.info("No data was successfully extracted from any of the documents.")

        # Display raw JSON for each document
        st.subheader("Raw JSON Output Per Document:")
        for i, doc_data in enumerate(st.session_state.all_extracted_documents):
            with st.expander(f"Raw JSON for {doc_data.get('originalFileName', f'Document {i+1}')}"):
                st.json(doc_data)