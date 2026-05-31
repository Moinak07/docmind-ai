# DocMind AI

DocMind AI is a Streamlit-based conversational RAG application that lets users upload PDF files and ask questions about their content. It uses LangChain, FAISS, HuggingFace embeddings, and Groq's LLM API to retrieve relevant PDF chunks and generate grounded answers.

The project is designed as a beginner-friendly but practical Retrieval-Augmented Generation app with PDF upload, chat history, streaming responses, session management, and retrieved-chunk debugging.

## Features

- Upload one or more PDF files
- Ask questions about uploaded PDF content
- Conversational RAG with chat history
- Streaming responses similar to ChatGPT
- FAISS vector store for document retrieval
- HuggingFace sentence-transformer embeddings
- Groq LLM integration
- Saved PDFs across browser refreshes
- Saved chat history by session ID
- Session management with a `Use session` button
- API key input with a `Use API key` button
- Collapsible retrieved-chunks debug panel
- Strict PDF-grounded prompting to reduce hallucination
- Separate frontend and backend files:
  - `app.py` for Streamlit UI
  - `fullrag.py` for RAG, LLM, PDF, and persistence logic

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq
- FAISS
- HuggingFace Embeddings
- Sentence Transformers
- PyPDF
- python-dotenv

## How It Works

1. The user uploads PDF files from the Streamlit sidebar.
2. PDFs are saved locally inside `data/pdfs/`.
3. PDF text is loaded using `PyPDFLoader`.
4. Text is split into chunks.
5. Chunks are converted into embeddings using `all-MiniLM-L6-v2`.
6. Embeddings are stored in a FAISS vector store.
7. When the user asks a question, the retriever finds relevant chunks.
8. The retrieved chunks are passed to the Groq LLM.
9. The LLM answers using only the retrieved PDF context.
10. The answer is streamed back to the Streamlit UI.

## Project Structure

```text
docmind-ai/
|
|-- app.py                  # Streamlit frontend
|-- fullrag.py              # RAG, LLM, embeddings, PDF, and persistence logic
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
|-- .gitignore              # Ignored local/private files
|
|-- data/                   # Local saved PDFs and chat history, ignored by Git
|   |-- pdfs/
|   `-- chat_histories.json
|
`-- .env                    # Local secrets, ignored by Git
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

`GROQ_API_KEY` is used for the Groq LLM.

`HF_TOKEN` is optional for many public HuggingFace models, but useful if your environment requires HuggingFace authentication.

Never upload `.env` to GitHub.

## Installation

Clone the repository:

```bash
git clone https://github.com/Moinak07/docmind-ai.git
cd docmind-ai
```

Create and activate a virtual environment.

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running The App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Usage

1. Enter your Groq API key in the sidebar.
2. Click `Use API key`.
3. Enter or keep the session name.
4. Click `Use session`.
5. Upload one or more PDF files.
6. Ask questions in the chat input.
7. Open `Retrieved chunks debug` to inspect which PDF chunks were retrieved.

## RAG Configuration

Current settings are defined in `fullrag.py`:

```python
DEFAULT_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

You can tune these values to improve retrieval quality.

## Important GitHub Safety Notes

The project uses `.gitignore` to avoid uploading private or generated files:

```gitignore
.env
venv/
data/
__pycache__/
*.pyc
temp.pdf
```

Do not push:

- API keys
- `.env`
- Uploaded PDFs
- Chat history
- Virtual environment folders

## Deployment Notes

When deploying to Streamlit Community Cloud or another platform:

1. Push only the safe project files to GitHub.
2. Add secrets through the deployment platform's secret manager.
3. Do not upload `.env`.
4. Make sure `requirements.txt` contains all required packages.

For Streamlit Community Cloud, add secrets in the app settings:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
HF_TOKEN = "your_huggingface_token_here"
```

## Limitations

- Scanned image-only PDFs may not work unless OCR is added.
- Very large PDFs may take time to embed.
- FAISS vector store is rebuilt when PDFs change.
- Answers depend on the quality of retrieved chunks.
- The app is designed for local/single-user usage by default.

## Possible Future Improvements

- Add OCR support for scanned PDFs
- Add source citations directly below every answer
- Add PDF preview
- Add user authentication
- Add persistent cloud storage
- Add model selector
- Add better evaluation for retrieval quality

## Author

Built by [Moinak07](https://github.com/Moinak07).
