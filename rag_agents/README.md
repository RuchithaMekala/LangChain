
# Advanced RAG Q&A Project

This project demonstrates an advanced Retrieval-Augmented Generation (RAG) Q&A system that integrates multiple data sources including Wikipedia, arXiv, specific documentation URLs, and PDFs.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

Makesure you have OpenAI API key with you in .env file. Maintain the file in your environment.

### Installation

1. Create a virtual environment:

   ```bash
   python3 -m venv ~/.venvs/rag_llm
   source ~/.venvs/rag_llm/bin/activate

2. Install all the requirements:

   ```bash
   pip install -r requirements.txt

3. Run the application:

   ```bash
   streamlit run application.py

