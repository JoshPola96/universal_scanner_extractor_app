# modules/llm_utils.py

import streamlit as st
import google.generativeai as genai
import json
import time
import os

# Set this to True to enable detailed terminal logging for debugging LLM output
DEBUG_LLM_LOGGING = True

def get_llm_model():
    if DEBUG_LLM_LOGGING:
        print("Starting get_llm_model function.")
    if "llm_model" not in st.session_state or st.session_state.llm_model is None:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY") # Check environment variables first
        
            if not api_key:
                # Fallback to Streamlit secrets if not found in environment variables
                if "GOOGLE_API_KEY" in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
                    
            st.session_state.llm_model = genai.GenerativeModel('gemma-3-27b-it')
            if DEBUG_LLM_LOGGING:
                print("Gemini model initialized successfully.")
        except KeyError:
            st.error("Google Gemini API Key not found. Please set it in Streamlit secrets as 'GOOGLE_API_KEY'.")
            st.session_state.llm_model = None
            if DEBUG_LLM_LOGGING:
                print("Error: Google Gemini API Key not found.")
        except Exception as e:
            st.error(f"Error initializing Gemini model: {e}. Please check your API key and model access.")
            st.session_state.llm_model = None
            if DEBUG_LLM_LOGGING:
                print(f"Error initializing Gemini model: {e}")
    if DEBUG_LLM_LOGGING and st.session_state.llm_model:
        print("Returning initialized LLM model.")
    elif DEBUG_LLM_LOGGING and st.session_state.llm_model is None:
        print("Returning None for LLM model (initialization failed).")
    return st.session_state.llm_model

def parse_numeric_value(value):
    """
    Parses a string value into a float, handling common numeric formats.
    """
    if DEBUG_LLM_LOGGING:
        print(f"Starting parse_numeric_value for value: '{value}'")
    if value is None:
        if DEBUG_LLM_LOGGING:
            print("   Input value is None. Returning None.")
        return None
    if isinstance(value, (int, float)):
        if DEBUG_LLM_LOGGING:
            print(f"   Input value is already numeric. Returning float: {float(value)}")
        return float(value)
    
    cleaned_str = str(value).replace(',', '').replace('$', '').replace('€', '').strip()
    try:
        parsed_float = float(cleaned_str)
        if DEBUG_LLM_LOGGING:
            print(f"   Successfully parsed '{value}' to float: {parsed_float}")
        return parsed_float
    except ValueError:
        if DEBUG_LLM_LOGGING:
            print(f"   Could not parse '{value}' to float. Returning None.")
        return None

def call_llm_with_retry(prompt, max_retries=5, initial_delay=1, task_description="LLM call"):
    """
    Calls the LLM model with retry logic for transient errors like rate limits.
    Includes a progress bar for better UX.
    """
    if DEBUG_LLM_LOGGING:
        print(f"Starting call_llm_with_retry for '{task_description}' (max_retries: {max_retries}, initial_delay: {initial_delay}).")
        # print(f"Prompt preview:\n{prompt[:200]}...") # Print first 200 chars of prompt

    llm_model = get_llm_model()
    if llm_model is None:
        if DEBUG_LLM_LOGGING:
            print(f"LLM model not available for '{task_description}'. Aborting.")
        return None

    delay = initial_delay
    progress_bar = st.progress(0, text=f"{task_description} (Attempt 1/{max_retries})...")
    
    for i in range(max_retries):
        if DEBUG_LLM_LOGGING:
            print(f"Attempt {i+1}/{max_retries} to call LLM for '{task_description}'...")
        
        progress_bar.progress((i + 1) / max_retries, text=f"{task_description} (Attempt {i+1}/{max_retries})...")
        
        try:
            response = llm_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up markdown code blocks if present
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()
                if DEBUG_LLM_LOGGING:
                    print("   Stripped '```json' markdown from response.")
            elif response_text.startswith("```") and response_text.endswith("```"):
                response_text = response_text[3:-3].strip()
                if DEBUG_LLM_LOGGING:
                    print("   Stripped generic '```' markdown from response.")
            
            if DEBUG_LLM_LOGGING:
                print(f"   LLM call for '{task_description}' successful on attempt {i+1}.")
                # print(f"   Response preview:\n{response_text[:200]}...")
            progress_bar.empty() # Clear the progress bar on success
            return response_text
        except Exception as e:
            st.warning(f"LLM call for '{task_description}' failed (Attempt {i+1}/{max_retries}): {e}")
            if "ResourceExhausted" in str(e) or "Too Many Requests" in str(e):
                st.info(f"Rate limit hit for '{task_description}'. Retrying in {delay} seconds...")
                if DEBUG_LLM_LOGGING:
                    print(f"   Rate limit error for '{task_description}'. Retrying in {delay}s.")
                time.sleep(delay)
                delay *= 2
            else:
                st.info(f"Non-rate limit error for '{task_description}'. Retrying anyway...")
                if DEBUG_LLM_LOGGING:
                    print(f"   Non-rate limit error for '{task_description}': {e}. Retrying in {delay}s.")
                time.sleep(delay)
                delay *= 2
    st.error(f"Failed to get LLM response for '{task_description}' after {max_retries} retries.")
    progress_bar.empty() # Clear the progress bar on failure
    if DEBUG_LLM_LOGGING:
        print(f"LLM call for '{task_description}' failed after {max_retries} retries. Returning None.")
    return None

def infer_document_schema(document_text):
    """
    Uses LLM to infer a JSON schema for the document based on its content.
    The schema should be general-purpose, identifying key-value pairs and line items.
    """
    if DEBUG_LLM_LOGGING:
        print("Starting infer_document_schema function.")
        print(f"Document text length for schema inference: {len(document_text)} characters.")

    schema_prompt = f"""
    You are an expert at identifying the structure and key fields of any document.
    Given the following document text, propose a JSON schema that describes the main
    information entities and their types.

    Identify prominent key-value pairs (e.g., "Invoice Number:", "Date:", "Total:"),
    and list-like structures (e.g., line items in invoices, or lists of products).

    For general key-value pairs, use descriptive camelCase names (e.g., 'documentTitle', 'dateOfIssue', 'referenceNumber', 'totalAmount').
    Use 'string' for text, 'float' for numbers, and 'boolean' for boolean values.
    For dates, use 'string' and indicate 'format: "YYYY-MM-DD"' in the description.

    **CRITICAL**: For list-like structures, especially those resembling line items, you MUST structure them as an array of objects
    using the following JSON Schema standard format. Each object in the array should have its properties defined under `properties`.

    Example of an array of objects schema for 'items':
    ```json
    "items": {{
        "type": "array",
        "items": {{
            "type": "object",
            "properties": {{
                "lineNumber": {{ "type": "integer" }},
                "quantity": {{ "type": "number" }},
                "description": {{ "type": "string" }},
                "unitPrice": {{ "type": "number" }},
                "netWorth": {{ "type": "number" }}
            }}
        }}
    }}
    ```
    You can infer other relevant properties for list items beyond this example (e.g., 'unit', 'vatPercentage', 'grossWorth').
    For numeric types in the schema, use "number" (which covers both floats and integers in JSON Schema).

    If a field or structure is not clearly present, do not include it in the schema, or mark its type as 'null'.
    Ensure the output is a valid JSON object.
    The JSON should ONLY contain the 'properties' part of a JSON schema object, and nothing else.
    DO NOT include any explanation or additional text outside the JSON object.

    Document Text:
    ---
    {document_text}
    ---

    Proposed JSON Schema (ONLY the 'properties' object):
    """
    response_text = call_llm_with_retry(schema_prompt, task_description="Inferring Schema")
    if response_text:
        try:
            parsed_response = json.loads(response_text)
            # The prompt is now strict about returning only the 'properties' object,
            # but we keep the 'if 'properties' in parsed_response' for robustness
            if 'properties' in parsed_response: 
                if DEBUG_LLM_LOGGING:
                    print("   Inferred schema wrapped in 'properties'. Extracting.")
                return parsed_response['properties']
            
            if DEBUG_LLM_LOGGING:
                print("   Inferred schema directly provides properties object.")
            return parsed_response 
        except json.JSONDecodeError:
            st.error(f"Error decoding inferred schema JSON: {response_text}")
            if DEBUG_LLM_LOGGING:
                print(f"   JSONDecodeError for inferred schema: {response_text}")
            return {}
    if DEBUG_LLM_LOGGING:
        print("   No response text for schema inference. Returning empty dict.")
    return {}

def extract_data_with_inferred_schema(document_text, inferred_schema):
    """
    Uses LLM to extract data based on a previously inferred schema.
    """
    if DEBUG_LLM_LOGGING:
        print("Starting extract_data_with_inferred_schema function.")
        print(f"Document text length for data extraction: {len(document_text)} characters.")
        print(f"Inferred schema received: {json.dumps(inferred_schema, indent=2)}")

    if not inferred_schema:
        st.warning("No schema provided for extraction. Skipping LLM extraction.")
        if DEBUG_LLM_LOGGING:
            print("   No inferred schema. Skipping extraction.")
        return {}

    field_descriptions = []
    for field, details in inferred_schema.items():
        # Ensure 'details' is a dictionary. If it's a string (e.g., "string"),
        # convert it to the expected dictionary format {"type": "string"}.
        if isinstance(details, str):
            if DEBUG_LLM_LOGGING:
                print(f"   Schema detail for '{field}' is a string '{details}'. Converting to dict for processing.")
            details = {"type": details}

        if details.get("type") == "array" and details.get("items", {}).get("type") == "object" and "properties" in details.get("items", {}):
            # Handle array of objects (like line items)
            item_properties = details["items"]["properties"]
            item_properties_str = ", ".join([f"{prop_name} ({prop_details.get('type', 'string')})"
                                             for prop_name, prop_details in item_properties.items()])
            field_descriptions.append(
                f"- `{field}` (array of objects, each with properties: {item_properties_str}). Description: {details.get('description', '')}"
            )
            if DEBUG_LLM_LOGGING:
                print(f"   Added array schema description for '{field}': {item_properties_str}")
        else:
            # Handle simple key-value pairs
            field_descriptions.append(
                f"- `{field}` ({details.get('type', 'string')}). Description: {details.get('description', '')}"
            )
            if DEBUG_LLM_LOGGING:
                print(f"   Added simple field schema description for '{field}': {details.get('type', 'string')}")

    required_fields_list = "\n".join(field_descriptions)
    if DEBUG_LLM_LOGGING:
        print(f"Generated required fields list for extraction prompt:\n{required_fields_list}")

    extraction_prompt = f"""
    You are an expert at extracting structured data from documents according to a specific JSON schema.
    Given the following document text, extract the data strictly according to the described JSON schema.
    
    IMPORTANT RULES FOR EXTRACTION:
    1.  **Strict JSON Output**: The output MUST be a valid JSON object.
    2.  **Schema Adherence**: Adhere strictly to the field names, types, and structure defined in the schema below.
    3.  **Null Values**: If a field or any attribute within a list item is not found or clearly identifiable in the document text, use `null` for its value. DO NOT guess or hallucinate values.
    4.  **Numeric Types**: Numeric values (e.g., 'float', 'integer', 'number') MUST be provided as actual numbers (e.g., 123.45), not strings ("123.45"). Remove any currency symbols ($, €, etc.) or commas (,) from numbers.
    5.  **Date Format**: Dates MUST be in `YYYY-MM-DD` format if identifiable. If only month/year or day/month/year is present, convert to `YYYY-MM-DD` (e.g., "June 2023" -> "2023-06-XX", "15/03/2023" -> "2023-03-15"). If not clear, use `null`.
    6.  **Boolean Values**: Boolean values MUST be `true` or `false` (lowercase).
    7.  **Arrays/Lists**: If a list-like structure (e.g., line items) is defined as an array in the schema but no items are found, the value for that field should be an empty array `[]`. If items are found, ensure they are objects adhering to their defined properties.
    8.  **No Explanations**: Do NOT include any introductory or concluding remarks, explanations, or text outside the JSON object.

    Required Fields and their Schema:
    {required_fields_list}

    Document Text:
    ---
    {document_text}
    ---

    JSON Output:
    """
    response_text = call_llm_with_retry(extraction_prompt, task_description="Extracting Data")
    if response_text:
        try:
            parsed_data = json.loads(response_text)
            if DEBUG_LLM_LOGGING:
                print("   Successfully parsed extracted data JSON.")
                # print(f"   Parsed Data: {json.dumps(parsed_data, indent=2)}")
            return parsed_data
        except json.JSONDecodeError:
            st.error(f"Error decoding extracted data JSON: {response_text}")
            if DEBUG_LLM_LOGGING:
                print(f"   JSONDecodeError for extracted data: {response_text}")
            return {}
    if DEBUG_LLM_LOGGING:
        print("   No response text for data extraction. Returning empty dict.")
    return {}

def self_correct_extracted_data(raw_extracted_data, document_text, inferred_schema, max_retries=2):
    """
    Uses LLM to review and self-correct previously extracted data against the inferred schema.
    """
    if DEBUG_LLM_LOGGING:
        print("Starting self_correct_extracted_data function.")
        print(f"Raw extracted data for self-correction: {json.dumps(raw_extracted_data, indent=2)}")
        print(f"Inferred schema for self-correction: {json.dumps(inferred_schema, indent=2)}")
    
    current_data = raw_extracted_data
    for i in range(max_retries):
        correction_prompt = f"""
        You are an AI assistant tasked with validating and correcting structured data extracted from a document against a given JSON schema.
        
        Review the 'CURRENT EXTRACTED DATA' provided below and compare it against the 'REQUIRED SCHEMA'.
        
        Identify any discrepancies, such as:
        - Incorrect data types (e.g., string instead of float for a numeric field).
        - Missing fields that should be present and are not `null` if they exist in the document.
        - Incorrect date formats (should be YYYY-MM-DD).
        - Numeric values containing currency symbols or commas.
        - Array structures not correctly formatted (e.g., line items not being a list of objects or missing properties within objects).
        
        If corrections are needed, generate a *complete* corrected JSON object that strictly adheres to the schema.
        If no corrections are needed, return the CURRENT EXTRACTED DATA as is.
        
        IMPORTANT RULES FOR CORRECTION:
        1.  **Strict JSON Output**: The output MUST be a valid JSON object.
        2.  **Schema Adherence**: The corrected JSON MUST strictly conform to the 'REQUIRED SCHEMA' in terms of field names, types, and structure.
        3.  **Null Values**: If information is genuinely not present in the original 'DOCUMENT TEXT', it should remain `null` or an empty array `[]` if it's an array type.
        4.  **Numeric/Date Formatting**: Apply the same strict formatting for numbers and dates as in the initial extraction instructions (floats/integers, YYYY-MM-DD).
        5.  **No Explanations**: Do NOT include any introductory or concluding remarks, explanations, or text outside the JSON object.

        REQUIRED SCHEMA:
        {json.dumps(inferred_schema, indent=2)}

        DOCUMENT TEXT (for context):
        ---
        {document_text}
        ---

        CURRENT EXTRACTED DATA (to be reviewed and corrected):
        ---
        {json.dumps(current_data, indent=2)}
        ---

        Corrected JSON Output (or the original if no corrections are needed):
        """
        
        corrected_response_text = call_llm_with_retry(
            correction_prompt, 
            task_description=f"Self-correcting data (Attempt {i+1}/{max_retries})", 
            max_retries=1 # Self-correction prompt itself should not retry internally
        )

        if corrected_response_text:
            try:
                corrected_data = json.loads(corrected_response_text)
                # Check if the data actually changed, otherwise stop
                if corrected_data == current_data:
                    if DEBUG_LLM_LOGGING:
                        print(f"   Self-correction on attempt {i+1} found no changes needed or made no effective changes. Stopping.")
                    return corrected_data
                else:
                    current_data = corrected_data
                    if DEBUG_LLM_LOGGING:
                        print(f"   Self-correction on attempt {i+1} made corrections. New data: {json.dumps(current_data, indent=2)}")
            except json.JSONDecodeError:
                st.warning(f"Self-correction LLM returned invalid JSON on attempt {i+1}. Trying again.")
                if DEBUG_LLM_LOGGING:
                    print(f"   Self-correction JSONDecodeError: {corrected_response_text}")
                # Wait before next retry for self-correction
                time.sleep(2) 
        else:
            st.warning(f"Self-correction LLM did not return a response on attempt {i+1}.")
            # Wait before next retry for self-correction
            time.sleep(2)

    if DEBUG_LLM_LOGGING:
        print(f"Finished self_correct_extracted_data after {max_retries} attempts. Final data: {json.dumps(current_data, indent=2)}")
    return current_data

def post_process_extracted_data(extracted_data, inferred_schema):
    """
    Applies post-processing to extracted data based on the inferred schema,
    e.g., converting numeric strings to floats.
    This function acts as a final safeguard after LLM extraction/self-correction.
    """
    if DEBUG_LLM_LOGGING:
        print("Starting post_process_extracted_data function.")
        print(f"Raw extracted data: {json.dumps(extracted_data, indent=2)}")
        print(f"Inferred schema used for post-processing: {json.dumps(inferred_schema, indent=2)}")

    processed_data = {}
    for key, value in extracted_data.items():
        schema_info = inferred_schema.get(key)
        # Ensure 'schema_info' is a dictionary for direct type access.
        # This mirrors the change in extract_data_with_inferred_schema for consistency.
        if isinstance(schema_info, str):
            if DEBUG_LLM_LOGGING:
                print(f"   Post-processing: Schema info for '{key}' is a string '{schema_info}'. Converting to dict.")
            schema_info = {"type": schema_info}

        if schema_info:
            if schema_info.get('type') == 'float' or schema_info.get('type') == 'number':
                processed_data[key] = parse_numeric_value(value)
                if DEBUG_LLM_LOGGING:
                    print(f"   Post-processing '{key}': Converted to float/number: {processed_data[key]}")
            elif schema_info.get('type') == 'array' and schema_info.get('items', {}).get('type') == 'object':
                processed_items = []
                if isinstance(value, list):
                    for item in value:
                        processed_item = {}
                        if isinstance(item, dict):
                            for prop_key, prop_value in item.items():
                                prop_schema_info = schema_info.get('items', {}).get('properties', {}).get(prop_key)
                                # Ensure 'prop_schema_info' is a dictionary for direct type access.
                                if isinstance(prop_schema_info, str):
                                    if DEBUG_LLM_LOGGING:
                                        print(f"     Post-processing: Item prop schema info for '{prop_key}' is a string '{prop_schema_info}'. Converting to dict.")
                                    prop_schema_info = {"type": prop_schema_info}
                                if prop_schema_info and (prop_schema_info.get('type') == 'float' or prop_schema_info.get('type') == 'number'):
                                    processed_item[prop_key] = parse_numeric_value(prop_value)
                                    if DEBUG_LLM_LOGGING:
                                        print(f"     Post-processing line item '{prop_key}': Converted to float/number: {processed_item[prop_key]}")
                                else:
                                    processed_item[prop_key] = prop_value
                            processed_items.append(processed_item)
                        else:
                            if DEBUG_LLM_LOGGING:
                                print(f"     Skipping non-dict item in array for '{key}': {item}")
                            processed_items.append(item) # Append as is if not a dict
                else:
                    if DEBUG_LLM_LOGGING:
                        print(f"   Expected array for '{key}' but got non-list type: {type(value)}. Assigning as is.")
                    processed_items = value # Assign as is if not a list
                processed_data[key] = processed_items
            else:
                processed_data[key] = value
                if DEBUG_LLM_LOGGING:
                    print(f"   No specific post-processing for '{key}'. Value: {value}")
        else:
            processed_data[key] = value
            if DEBUG_LLM_LOGGING:
                print(f"   No schema info for '{key}'. Value: {value}")
    
    if DEBUG_LLM_LOGGING:
        print(f"Finished post_process_extracted_data. Processed data: {json.dumps(processed_data, indent=2)}")
    return processed_data