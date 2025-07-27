# README.md

# Persona-Driven Document Intelligence (Hackathon Round 1B)

This project is a solution for the "Connecting the Dots" Hackathon, Round 1B. It analyzes a collection of PDF documents to extract the most relevant sections based on a specific user persona and their job-to-be-done.

## Approach

The solution uses a two-stage pipeline:

1.  **Stage 1: Structural Parsing (1A Model)**
    *   The system first processes each PDF using a custom parser (`parser_1a.py`).
    *   It leverages the **PyMuPDF** library to extract detailed text span information, including text content, font size, font weight (bold), and position.
    *   These attributes are used as features for a pre-trained **LightGBM** machine learning model (`heading_model_large8.txt`).
    *   This model predicts whether each text span is a title, a heading (H1, H2, H3), or regular body text.
    *   The output is a structured JSON outline of the document, which forms the basis for the next stage.

2.  **Stage 2: Persona-Based Analysis (1B Analyzer)**
    *   The `analyzer_1b.py` module takes the structured outlines from Stage 1 and uses them to intelligently segment the documents into meaningful "chunks" (i.e., the text content under each heading).
    *   It combines the `persona` and `job_to_be_done` from the input JSON into a single query.
    *   **scikit-learn's TfidfVectorizer** and **Cosine Similarity** are used to perform a semantic search. It calculates a relevance score between the user query and every text chunk from all documents.
    *   The chunks are ranked by this score, and the top 15 are selected as the most relevant sections.
    *   For the "Sub-section Analysis", the same TF-IDF process is repeated at a granular level to find the *single most relevant sentence* within each of the top-ranked chunks.

## Libraries and Models

*   **Python 3.10**
*   **Libraries**: `PyMuPDF`, `LightGBM`, `scikit-learn`, `pandas`, `numpy`, `nltk`
*   **Model**: A pre-trained LightGBM classification model (`.txt` format) is used for heading detection.

## How to Build and Run

### Prerequisites

*   Docker must be installed and running.

### Instructions

1.  **Place Files**:
    *   Place your trained 1A model (e.g., `heading_model_large8.txt`) inside the `/models` directory.
    *   Create a local folder named `test_input` in the project root.
    *   Place `challenge1b_input.json` and all required PDF documents inside the `test_input` folder.
    *   Create an empty local folder named `test_output` in the project root.

2.  **Build the Docker Image**:
    Open a terminal in the project root directory (`adobe_hackathon_1b/`) and run:
    ```sh
    docker build --platform linux/amd64 -t mysolution:latest .
    ```

3.  **Run the Container**:
    Execute the following command. It will mount your local test folders into the container, run the analysis without network access, and save the result back to your `test_output` folder.
    ```sh
    docker run --rm -v "%cd%\test_input":/app/input -v "%cd%\test_output":/app/output --network none mysolution:latest
    ```

4.  **Check the Output**:
    After the run completes, the file `challenge1b_final_output.json` will appear in your local `test_output` directory.