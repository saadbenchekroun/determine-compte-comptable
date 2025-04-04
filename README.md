# Semantic Search in Moroccan Chart of Accounts (Plan Comptable Marocain)
## Description

This project provides a Google Colab notebook demonstrating how to implement a powerful semantic search engine for the Moroccan Chart of Accounts (Plan Comptable Marocain - PCM).

The standard PCM contains numerous account descriptions (`Nomenclature du compte`) that are very similar, making simple keyword matching prone to errors. This notebook utilizes **Sentence Transformers** based on deep learning models to understand the *meaning* behind user queries and find the most relevant account entry, even when the phrasing differs slightly.

The goal is to achieve high accuracy in matching user input to the correct account number (`N° de compte`).

##  Features

*   **Semantic Understanding:** Goes beyond simple keywords to match based on meaning.
*   **Robust Matching:** Handles variations in phrasing, synonyms, and minor typos better than traditional methods.
*   **Confidence Scoring:** Provides a cosine similarity score indicating the confidence of the match.
*   **Exact Match Priority:** Checks for exact matches first for guaranteed precision when applicable.
*   **Pre-trained Models:** Leverages powerful, pre-trained language models (e.g., multilingual MPNet) fine-tuned for semantic similarity.
*   **Easy to Use:** Implemented as a step-by-step Google Colab notebook.

##  Technology Stack

*   **Python 3**
*   **Google Colab** (or any environment supporting Jupyter notebooks)
*   **Pandas:** For data loading and manipulation.
*   **Sentence-Transformers:** For generating semantic embeddings and calculating similarity.
*   **PyTorch** (or TensorFlow): As the backend deep learning framework for Sentence-Transformers.

## Prerequisites

You need the Moroccan Chart of Accounts in a CSV format. The notebook assumes a file named:

*   `Plan-comptable-Maroc-Excel.csv`

This file **must** contain at least the following columns:
*   `N° de compte` (Account Number)
*   `Nomenclature du compte` (Account Description/Name)

*(The provided notebook includes an upload step for this file.)*

##  Getting Started / Usage

1.  **Open the Notebook:** Click the "Open In Colab" badge above or manually upload the `Plan_Comptable_Semantic_Search.ipynb` file to Google Colab.
2.  **Enable GPU (Recommended):** For faster embedding generation, go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator in Colab.
3.  **Upload Data:** Run the cell under **"3. Chargement des données"**. You will be prompted to upload your `Plan-comptable-Maroc-Excel.csv` file.
4.  **Run All Cells:** Execute the notebook cells sequentially from top to bottom.
    *   Installs necessary libraries.
    *   Loads the language model (this might take a minute on the first run).
    *   Generates embeddings for all account descriptions in your CSV (this is the most time-consuming step).
    *   Defines the search function.
    *   Runs example queries to demonstrate the search functionality.
5.  **Interpret Results:** The demo cell will print the query, followed by the best matching account(s) found, including their number, nomenclature, and similarity score. Matches below the defined threshold will be discarded.

##  How it Works

1.  **Load Data:** The chart of accounts is loaded from the CSV file.
2.  **Load Model:** A pre-trained Sentence Transformer model (suitable for French/multilingual text) is loaded.
3.  **Generate Embeddings:** The model converts each account description (`Nomenclature du compte`) into a high-dimensional vector (embedding) that captures its semantic meaning. These embeddings are stored.
4.  **Query Processing:**
    *   **Exact Match:** The system first checks if the user's query exactly matches any account description (case-insensitive). If found, this is returned with a perfect score (1.0).
    *   **Semantic Search:** If no exact match exists, the user's query is also converted into an embedding using the *same* model.
    *   **Similarity Calculation:** The cosine similarity is calculated between the query embedding and all pre-computed account embeddings. Cosine similarity measures the orientation (angle) between vectors, indicating semantic closeness.
    *   **Ranking & Filtering:** Matches are ranked by their similarity score. Only matches exceeding a predefined `threshold` are considered valid results.
5.  **Return Results:** The top N matching accounts (above the threshold) are returned with their details and scores.

##  Configuration & Customization

You can easily adjust the following in the notebook:

*   **`model_name` (Cell 4):** Try different pre-trained models from the `sentence-transformers` library for potentially better performance (e.g., models based on CamemBERT or FlauBERT if available and suitable).
*   **`threshold` (Cell 6 & 7):** Adjust the minimum cosine similarity score required for a match to be considered valid. Higher values mean stricter matching (more precision, less recall), lower values mean looser matching (less precision, more recall).
*   **`top_n` (Cell 6 & 7):** Change the maximum number of top results to return for each query.
*   **Example Queries (Cell 7):** Modify or add your own test queries.

##  Potential Improvements & TODO

*   **Advanced Indexing (Faiss):** For very large charts of accounts, implement Facebook AI Similarity Search (Faiss) for significantly faster nearest neighbor search instead of linear comparison.
*   **Model Fine-tuning:** Collect pairs of (typical user query, correct account number) and fine-tune the Sentence Transformer model specifically for this accounting domain for potentially higher accuracy.
*   **Context Integration:** Experiment with incorporating `Classe` and `Rubrique` information into the embeddings (e.g., by prepending them to the `Nomenclature du compte` before encoding) or using them as filters/re-ranking criteria.
*   **User Interface:** Build a simple web interface (e.g., using Flask or Streamlit) around the search function.
*   **Disambiguation:** If multiple accounts have high similarity scores, implement logic to ask the user for clarification or present the top options clearly.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
