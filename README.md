This project demonstrates how to load a document, split it into chunks, embed it using OpenAI embeddings, store it in a Chroma vector database, and create a simple Streamlit app for question answering based on the document content.
Prerequisites

    Python 3.7+
    pip
    An OpenAI API key
    A .env file with your OpenAI API key (OPENAI_API_KEY)

## Installation

Clone the repository:

  bash

    git clone <repository-url>
    cd <repository-directory>

## Install the required packages:

  bash

    pip install -r requirements.txt

Create a .env file in the root directory and add your OpenAI API key:

  plaintext

    OPENAI_API_KEY=your_openai_api_key

### Usage

  Place the PDF file you want to load in the data/ directory and update the script with the correct filename if necessary.

  Run the script to process the document and start the Streamlit app:

  bash

    streamlit run app.py
