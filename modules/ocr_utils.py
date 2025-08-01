# modules/ocr_utils.py

import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR
from math import sin, cos, radians

# Set this to True to enable detailed terminal logging for debugging OCR output
DEBUG_OCR_LOGGING = True

# Initialize PaddleOCR model globally for caching efficiency
def get_paddle_ocr_model():
    # Only initialize the model if it's currently None in session state
    if st.session_state.paddle_ocr_model is None:
        with st.spinner("Initializing PaddleOCR model (first time might download models)..."):
            try:
                if DEBUG_OCR_LOGGING:
                    print("Attempting to initialize PaddleOCR model...")
                st.session_state.paddle_ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
                if DEBUG_OCR_LOGGING:
                    print("PaddleOCR model initialized successfully.")
            except Exception as e:
                if DEBUG_OCR_LOGGING:
                    print(f"Error initializing PaddleOCR: {e}")
                st.error(f"Error initializing PaddleOCR: {e}. Check Dockerfile dependencies or network connectivity for model download.")
                st.session_state.paddle_ocr_model = None
    return st.session_state.paddle_ocr_model

def deskew_image(image):
    """
    Detects and corrects the skew angle of an image.
    Handles calculating new dimensions to prevent cropping after rotation.
    Args:
        image (np.array): Grayscale image.
    Returns:
        np.array: Deskewed image.
    """
    if DEBUG_OCR_LOGGING:
        print("Starting deskew_image function.")

    # Find coordinates of all non-zero pixels (i.e., text/lines)
    coords = np.column_stack(np.where(image > 0))

    if len(coords) < 2: # Need at least 2 points for minAreaRect
        if DEBUG_OCR_LOGGING:
            print(f"Not enough non-zero pixels ({len(coords)}) to deskew. Returning original image.")
        return image

    try:
        # Get the rotated bounding box of these points
        # rect is ((center_x, center_y), (width, height), angle)
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        
        # The angle from minAreaRect is in the range [-90, 0)
        # Adjust the angle to be in a more intuitive range based on text orientation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if DEBUG_OCR_LOGGING:
            print(f"Detected skew angle: {angle} degrees.")

        # Calculate new image dimensions to prevent cropping after rotation
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        angle_rad = radians(angle)
        
        # Calculation for new_w and new_h to prevent cropping
        new_w = int(w * abs(cos(angle_rad)) + h * abs(sin(angle_rad)))
        new_h = int(w * abs(sin(angle_rad)) + h * abs(cos(angle_rad)))

        # Get the rotation matrix, adjusting translation to center the image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation, filling new areas with white
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        if DEBUG_OCR_LOGGING:
            print(f"Image deskewed successfully. Original size: ({w},{h}), New size: ({new_w},{new_h}).")
        return rotated
    except Exception as e:
        if DEBUG_OCR_LOGGING:
            print(f"Error during deskewing: {e}. Returning original image.")
        return image


def preprocess_image(image_bytes):
    """
    Applies comprehensive image pre-processing using OpenCV for better OCR results.
    - Converts to grayscale
    - Applies Gaussian blur for noise reduction
    - Binarization (thresholding)
    - Deskewing
    """
    if DEBUG_OCR_LOGGING:
        print("Starting preprocess_image function.")

    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode image bytes for pre-processing. Check image format or corruption.")
        if DEBUG_OCR_LOGGING:
            print("Failed to decode image bytes.")
        return None

    if DEBUG_OCR_LOGGING:
        print("Image decoded. Applying grayscale, blur, binarization...")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding for binarization
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Deskew the binarized image
    deskewed_img = deskew_image(thresh)

    if DEBUG_OCR_LOGGING:
        print("Image pre-processing complete.")
    return deskewed_img

def perform_ocr_on_image(image_bytes):
    """
    Performs OCR on the pre-processed image using PaddleOCR.
    Returns a list of dicts with 'text' and 'bbox'.
    """
    if DEBUG_OCR_LOGGING:
        print("Starting perform_ocr_on_image function.")

    ocr_engine = get_paddle_ocr_model()
    if ocr_engine is None:
        if DEBUG_OCR_LOGGING:
            print("OCR engine not initialized. Aborting OCR.")
        return []

    # Use a spinner for the pre-processing step within OCR
    with st.spinner("Image pre-processing for OCR..."):
        processed_img = preprocess_image(image_bytes)
    
    if processed_img is None:
        if DEBUG_OCR_LOGGING:
            print("Pre-processed image is None. Aborting OCR.")
        return []

    try:
        if DEBUG_OCR_LOGGING:
            print("Calling PaddleOCR engine.ocr()...")
        
        # Convert the single-channel grayscale image to a 3-channel BGR image
        processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

        # Use a spinner for the actual OCR process
        with st.spinner("Running PaddleOCR..."):
            result = ocr_engine.ocr(processed_img_bgr) 
        
        if DEBUG_OCR_LOGGING:
            print(f"PaddleOCR raw result type: {type(result)}")
            if result:
                print(f"PaddleOCR raw result (first element): {result[0] if len(result)>0 else 'N/A'}")
            else:
                print("PaddleOCR returned an empty result.")

        ocr_words = []
        # The new structure seems to be result[0] containing dict with 'rec_polys', 'rec_texts', 'rec_scores'
        if result and len(result) > 0 and isinstance(result[0], dict):
            page_data = result[0]
            text_boxes = page_data.get('rec_polys', [])
            texts = page_data.get('rec_texts', [])
            scores = page_data.get('rec_scores', [])

            if DEBUG_OCR_LOGGING:
                print(f"Processing {len(texts)} detected words/lines from rec_texts.")

            # Iterate through the recognized texts and their corresponding bounding boxes and scores
            for i in range(len(texts)):
                try:
                    bbox_coords = text_boxes[i]
                    text = texts[i]
                    confidence = scores[i]

                    # Ensure bbox_coords is a list/tuple of 4 points
                    if not isinstance(bbox_coords, np.ndarray) or bbox_coords.shape != (4, 2):
                        if DEBUG_OCR_LOGGING:
                            print(f"   Skipping malformed bbox_coords[{i}] (wrong shape or type): {bbox_coords}")
                        continue

                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    
                    # Check for empty x_coords or y_coords before min/max
                    if not x_coords or not y_coords:
                        if DEBUG_OCR_LOGGING:
                            print(f"   Skipping bbox[{i}] with no coordinates: {bbox_coords}")
                        continue

                    # Bbox format: [x_min, y_min, x_max, y_max]
                    bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]

                    if text.strip() and confidence > 0.5: # Filter by confidence and empty text
                        ocr_words.append({
                            'text': text.strip(),
                            'bbox': bbox,
                            'confidence': float(confidence)
                        })
                    else:
                        if DEBUG_OCR_LOGGING:
                            print(f"   Skipping word[{i}] due to empty text or low confidence ({confidence}): '{text}'")
                except Exception as inner_e:
                    if DEBUG_OCR_LOGGING:
                        print(f"   Error processing individual OCR result {i}: {inner_e}")
                    continue
        elif result and len(result) > 0 and isinstance(result[0], list): # Fallback for older PaddleOCR result format
            if DEBUG_OCR_LOGGING:
                print("PaddleOCR result appears to be in an older list of lists format.")
            for line in result[0]:
                if len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) == 2:
                    bbox_coords = line[0]
                    text = line[1][0]
                    confidence = line[1][1]

                    if not isinstance(bbox_coords, list) or len(bbox_coords) != 4:
                        if DEBUG_OCR_LOGGING:
                            print(f"   Skipping malformed bbox_coords (older format): {bbox_coords}")
                        continue

                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                    
                    if text.strip() and confidence > 0.5:
                        ocr_words.append({
                            'text': text.strip(),
                            'bbox': bbox,
                            'confidence': float(confidence)
                        })
                    else:
                        if DEBUG_OCR_LOGGING:
                            print(f"   Skipping word (older format) due to empty text or low confidence ({confidence}): '{text}'")
                else:
                    if DEBUG_OCR_LOGGING:
                        print(f"   Skipping malformed line (older format): {line}")

    except Exception as e:
        st.error(f"Error during PaddleOCR execution: {e}. Please ensure the image is valid and model is loaded correctly.")
        if DEBUG_OCR_LOGGING:
            print(f"Error during PaddleOCR execution: {e}")
        return []

    if DEBUG_OCR_LOGGING:
        print(f"OCR completed. Extracted {len(ocr_words)} words.")
    return ocr_words

def reconstruct_text_from_ocr(ocr_words, x_tolerance=10, y_tolerance=5):
    """
    Reconstructs coherent text from OCR word detections, attempting to group words
    into lines and paragraphs based on their bounding box positions.
    Sorts by y-coordinate (top to bottom) then by x-coordinate (left to right).
    """
    if DEBUG_OCR_LOGGING:
        print(f"Starting reconstruct_text_from_ocr with {len(ocr_words)} words.")

    if not ocr_words:
        if DEBUG_OCR_LOGGING:
            print("No OCR words to reconstruct.")
        return ""

    # Sort words primarily by top y-coordinate, then by left x-coordinate
    sorted_words = sorted(ocr_words, key=lambda w: (w['bbox'][1], w['bbox'][0]))

    document_lines = []
    current_line = []
    
    if DEBUG_OCR_LOGGING:
        print("Attempting to group words into lines.")

    for i, word_info in enumerate(sorted_words):
        if not current_line:
            current_line.append(word_info)
        else:
            # Check if the word belongs to the current line
            # It belongs if its y-coordinate is within a tolerance of the current line's average y
            # And its x-coordinate is to the right of the last word in the line
            
            # Calculate average y of current line for comparison
            current_line_y_min = min(w['bbox'][1] for w in current_line)
            current_line_y_max = max(w['bbox'][3] for w in current_line)
            current_line_center_y = (current_line_y_min + current_line_y_max) / 2

            word_y_min, word_x_min = word_info['bbox'][1], word_info['bbox'][0]
            last_word_x_max = current_line[-1]['bbox'][2]

            # Condition 1: Y-overlap/proximity (word's vertical span should overlap or be close to line's vertical span)
            y_overlap = max(0, min(current_line_y_max, word_info['bbox'][3]) - max(current_line_y_min, word_y_min))
            min_height = min(current_line_y_max - current_line_y_min, word_info['bbox'][3] - word_info['bbox'][1])
            
            # Use relative y-position and overlap for more robust line detection
            is_y_aligned = (abs(word_y_min - current_line_y_min) <= y_tolerance) or \
                           (word_info['bbox'][3] <= current_line_y_max + y_tolerance and word_y_min >= current_line_y_min - y_tolerance)
            
            # Condition 2: X-position (word should be to the right, possibly with a small gap tolerance)
            is_x_sequential = word_x_min >= last_word_x_max - x_tolerance # Allow small overlap or immediate adjacency

            if is_y_aligned and is_x_sequential:
                current_line.append(word_info)
            else:
                # If not part of the current line, start a new line
                document_lines.append(current_line)
                current_line = [word_info]
    
    if current_line:
        document_lines.append(current_line)

    final_text_lines = []
    for line_words in document_lines:
        # Sort words within a line purely by x-coordinate to ensure correct reading order
        line_words_sorted_by_x = sorted(line_words, key=lambda w: w['bbox'][0])
        final_text_lines.append(" ".join([w['text'] for w in line_words_sorted_by_x]))
    
    reconstructed_text = "\n".join(final_text_lines)
    
    if DEBUG_OCR_LOGGING:
        print(f"Reconstruction complete. Generated {len(final_text_lines)} lines.")
    return reconstructed_text