# Rag-App
implements a Multi-Agent Knowledge Assistant using Retrieval-Augmented Generation (RAG), a calculator, and a dictionary tool, all served via Streamlit:

---

# 🧠 Multi-Agent Knowledge Assistant

This project is a **Streamlit-based web application** that provides an intelligent assistant capable of answering questions using:

* **Retrieval-Augmented Generation (RAG)** for document-based queries
* **Calculator Tool** for math and logical expressions
* **Dictionary Tool** for explaining AI/ML-related terms

---

## 🚀 Features

* 📄 **Document Ingestion**: Upload `.txt`, `.pdf`, or `.docx` files and automatically chunk them for semantic search.
* 🔍 **Semantic Retrieval**: Uses `sentence-transformers` and FAISS for efficient similarity search.
* 🤖 **LLM Integration**: Uses OpenAI's GPT models to generate intelligent responses.
* 🧮 **Calculator Tool**: Detects and solves math expressions.
* 📘 **Dictionary Tool**: Explains technical terms like RAG, embeddings, LLM, etc.
* 🧠 **Multi-Agent Orchestration**: Automatically selects the best tool (RAG, Calculator, or Dictionary) for each query.

---

## 🧰 Tech Stack

* **Python 3.8+**
* **Streamlit** – for the web interface
* **NLTK** – for sentence tokenization
* **PyPDF2 / python-docx** – for file parsing
* **sentence-transformers** – for embedding generation
* **FAISS** – for vector similarity search
* **OpenAI GPT-3.5 / GPT-4** – for intelligent answers
* **dotenv** – to securely load your OpenAI API key

---

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/knowledge-assistant.git
   cd knowledge-assistant
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment:**

   Create a `.env` file and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the app:**

   ```bash
   streamlit run main.py
   ```

---

## 📁 File Structure

```
├── main.py               # Main Streamlit app
├── requirements.txt      # Required Python packages
├── .env                  # (Not included) Your OpenAI API key
└── nltk_data/            # NLTK tokenizer data
```

---

## 📚 Sample Documents

The app comes preloaded with sample documents about:

* Retrieval-Augmented Generation (RAG)
* LLM Fundamentals
* Vector Databases
* Prompt Engineering
* Multi-Agent Systems

You can also upload your own documents to expand the assistant's knowledge base.

---

## 🧪 How It Works

1. **Chunking**:

   * Uploaded documents are split into overlapping chunks for better context coverage.
2. **Embedding & Indexing**:

   * Chunks are converted into embeddings using a SentenceTransformer model.
   * These are indexed with FAISS for fast similarity search.
3. **Query Handling**:

   * The app selects the appropriate tool (Calculator, Dictionary, or RAG) based on the query.
   * If RAG is selected, the most relevant chunks are retrieved and passed to an OpenAI model to generate an answer.

---

## 📌 Example Queries

* **Calculator**:
  `What is 23 + 56 * 2?`

* **Dictionary**:
  `Define RAG`
  `What does vector mean in machine learning?`

* **RAG**:
  `How does retrieval-augmented generation improve LLMs?`
  `What is FAISS used for in AI?`

---

## 🛠️ TODOs

* Add persistent file storage for uploaded documents.
* Improve UI/UX styling with custom components.
* Add user authentication (if deployed publicly).
* Add more tools like weather, currency converter, or web search.

