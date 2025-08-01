# Universal Document Scanner (LLM Powered)

## 📄 Overview

This project is a Streamlit web application that serves as a powerful, universal document scanner and data extractor. It leverages advanced Optical Character Recognition (OCR) and a Large Language Model (LLM) to intelligently process documents, infer their underlying structure, and extract key information.

Unlike traditional tools that require pre-defined templates, this application is highly adaptable. It can process a variety of document types, from invoices and forms to reports and receipts, by dynamically inferring a JSON schema from the document's content.

-----

## ✨ Features

  * **Universal Document Processing**: Automatically handles various document types and layouts.
  * **Image Pre-processing**: Uses OpenCV to enhance image quality with techniques like grayscale conversion, blurring, binarization, and **deskewing** for optimal OCR accuracy.
  * **Robust OCR**: Integrates **PaddleOCR**, a state-of-the-art OCR engine, for highly accurate text extraction.
  * **LLM-Powered Schema Inference**: A **Google Gemini/Gemma LLM** analyzes the OCR output from the first document to generate a structured JSON schema.
  * **Interactive Schema Confirmation**: Provides an editable interface for users to review and confirm the inferred schema before a batch extraction process begins.
  * **Batch Processing**: Efficiently processes multiple documents from a `.zip` archive using the confirmed schema.
  * **Structured Data Output**: Presents the extracted data in a clean, navigable pandas DataFrame.
  * **Data Export**: Allows users to download the extracted data as a CSV file.
  * **Docker Support**: Includes a `Dockerfile` for easy containerization, ensuring a consistent and reproducible environment.

-----

## ⚠️ **WARNING** ⚠️

This application is a **demonstration** of this powerful idea/concept and is not designed for secure, production-level use. Uploaded images and extracted data are processed in memory and will be cleared when you upload a new document or reload the page.

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.10+
  * Docker (if you choose the containerized approach)
  * A Google Gemini API key.

### Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/your-project-repo.git
    cd universal-document-scanner
    ```

2.  **Configure API Key**
    Create a `.streamlit` folder in your project's root directory and inside it, create a `secrets.toml` file.

    ```toml
    # .streamlit/secrets.toml
    GOOGLE_API_KEY="your_api_key_here"
    ```

    Replace `"your_api_key_here"` with your actual Google Gemini API key.

3.  **Installation & Running (Local)**

      * Create a virtual environment and activate it:
        ```bash
        python -m venv .venv
        # On Windows:
        .venv\Scripts\activate
        # On macOS/Linux:
        source .venv/bin/activate
        ```
      * Install the required dependencies:
        ```bash
        pip install -r requirements.txt
        ```
      * Run the Streamlit application:
        ```bash
        streamlit run app.py
        ```

4.  **Installation & Running (Docker)**

      * Make sure Docker is running on your system.
      * Build the Docker image. This may take a few minutes as it downloads the PaddleOCR models.
        ```bash
        docker build -t universal-scanner .
        ```
      * Run the container, making sure to pass your API key as a build argument:
        ```bash
        docker run -p 8501:8501 universal-scanner
        ```
      * The application will be accessible in your browser at `http://localhost:8501`.

-----

## 📝 Usage

The application follows a simple, three-step process:

1.  **Upload & Preview**: Upload a single image (JPG, PNG) or a ZIP file containing multiple images. The app will display a preview of the first document.

2.  **Infer & Confirm Schema**: Click the **"Infer Schema from First Document"** button. The application will perform OCR and use the LLM to generate a JSON schema. This schema will appear in an editable text area. Review and, if necessary, modify the schema to match your data requirements.

3.  **Extract Data**: Once you are satisfied with the schema, click **"Confirm Schema and Start Extraction"**. The app will use this schema to process all uploaded documents, extract the data, and display the results in a structured DataFrame. You can then download the data as a CSV file.

-----

## 📦 Project Structure

  * `app.py`: The main Streamlit application file that orchestrates the entire workflow and handles the user interface.
  * `modules/ocr_utils.py`: Contains all OCR-related functions, including image preprocessing, text extraction, and text reconstruction.
  * `modules/llm_utils.py`: Contains all LLM-related functions for schema inference, data extraction, and post-processing.
  * `requirements.txt`: Lists all Python dependencies required for the project.
  * `Dockerfile`: Defines the Docker image for a containerized, production-ready deployment.
  * `.streamlit/secrets.toml`: (Local development only) Stores your confidential API keys.

-----

## 🤝 Contributing

Contributions are welcome\! If you find a bug or have a feature request, please open an issue. If you'd like to contribute code, please submit a pull request.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.