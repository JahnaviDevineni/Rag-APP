# core_packages.py
import os
import re
import math
import nltk
import numpy as np
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai

# For document processing
from nltk.tokenize import sent_tokenize
from io import BytesIO
import PyPDF2
from docx import Document

# For embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss

# Configure NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure Gemini
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

class TextChunk:
    """Represents a chunk of text from a document."""
    
    def __init__(self, text: str, doc_name: str, chunk_id: int):
        self.text = text
        self.doc_name = doc_name
        self.chunk_id = chunk_id
        self.embedding = None
    
    def __str__(self):
        return f"Chunk {self.chunk_id} from {self.doc_name}: {self.text[:50]}..."

class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, uploaded_file) -> str:
        """Load document content from Streamlit file_uploader object."""
        content = ""
        
        if uploaded_file.type == "text/plain":
            content = uploaded_file.getvalue().decode("utf-8")
        
        elif uploaded_file.type == "application/pdf":
            with BytesIO(uploaded_file.getvalue()) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(uploaded_file.getvalue()))
            content = "\n".join([para.text for para in doc.paragraphs])
        
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.type}")
        
        return content
    
    def chunk_text(self, text: str, doc_name: str) -> List[TextChunk]:
        """Split text into overlapping chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(chunk_text, doc_name, chunk_id))
                chunk_id += 1
                
                overlap_size = 0
                while current_chunk and overlap_size < self.chunk_overlap:
                    overlap_size += len(current_chunk[0].split())
                    if overlap_size > self.chunk_overlap:
                        break
                    current_chunk.pop(0)
                
                current_size = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(chunk_text, doc_name, chunk_id))
        
        return chunks

class VectorStore:
    """Manages document embeddings and similarity search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
    
    def add_chunks(self, chunks: List[TextChunk]):
        """Add document chunks to the vector store."""
        for chunk in chunks:
            chunk.embedding = self.model.encode(chunk.text)
            self.chunks.append(chunk)
        
        embeddings = np.array([chunk.embedding for chunk in self.chunks]).astype('float32')
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        if len(self.chunks) > 0:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 3) -> List[TextChunk]:
        """Find the most similar chunks to the query."""
        if not self.chunks or self.index is None:
            return []
        
        query_embedding = self.model.encode(query).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        return [self.chunks[idx] for idx in indices[0]]

class Tool:
    """Base class for tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def can_handle(self, query: str) -> bool:
        raise NotImplementedError
    
    def execute(self, query: str) -> str:
        raise NotImplementedError

class CalculatorTool(Tool):
    """Tool for performing calculations."""
    
    def __init__(self):
        super().__init__("Calculator", "Performs mathematical calculations")
        self.patterns = [
            r'(\d+\s*[\+\-\*\/\^]\s*\d+)',
            r'(what is|calculate|compute|solve|evaluate)\s+(.+)',
            r'(\d+\s*[\+\-\*\/\^][\d\s\+\-\*\/\^]*)',
        ]
    
    def can_handle(self, query: str) -> bool:
        query = query.lower()
        calc_keywords = ["calculate", "computation", "math", "sum", "add", "subtract", 
                         "multiply", "divide", "squared", "cubed", "power", "root", 
                         "percentage", "percent"]
        
        if any(keyword in query for keyword in calc_keywords):
            return True
        
        return any(re.search(pattern, query) for pattern in self.patterns)
    
    def execute(self, query: str) -> str:
        query = query.lower()
        expression = ""
        
        for pattern in self.patterns:
            match = re.search(pattern, query)
            if match:
                expression = match.group(2 if "group(2)" in str(match.groups()) else 1)
                break
        
        if not expression:
            expression = re.sub(r'[^0-9\+\-\*\/\^\(\)\.\s]', '', query)
        
        expression = re.sub(r'[x÷]', lambda x: {'x': '*', '÷': '/'}[x.group()], expression)
        expression = expression.replace('^', '**')
        
        try:
            result = eval(expression, {"__builtins__": None}, {
                "abs": abs, "round": round, "max": max, "min": min,
                "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan, "log": math.log, "log10": math.log10,
                "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
                "pi": math.pi, "e": math.e
            })
            return f"Calculator result: {expression} = {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

class DictionaryTool(Tool):
    """Tool for looking up word definitions."""
    
    def __init__(self):
        super().__init__("Dictionary", "Looks up word definitions")
        self.dictionary = {
            "rag": "Retrieval-Augmented Generation...",
            "llm": "Large Language Model...",
            # ... (keep your existing dictionary entries)
        }
    
    def can_handle(self, query: str) -> bool:
        query = query.lower()
        definition_keywords = ["define", "definition", "what is", "what are", 
                               "meaning of", "explain the term"]
        
        return any(keyword in query and any(word in query for word in self.dictionary)
                   for keyword in definition_keywords)
    
    def execute(self, query: str) -> str:
        query = query.lower()
        for word in self.dictionary:
            if word in query:
                return f"Dictionary definition for '{word}': {self.dictionary[word]}"
        return "Definition not found."

class LLMService:
    """Service for interacting with Gemini models"""
    
    def __init__(self, model_name: str = "gemini-pro"):
        self.model = genai.GenerativeModel(model_name)
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

class RAGTool(Tool):
    def __init__(self, vector_store: VectorStore, llm_service):
        super().__init__("RAG", "Retrieves information from documents using RAG")
        self.vector_store = vector_store
        self.llm_service = llm_service
    
    def execute(self, query: str) -> str:
        relevant_chunks = self.vector_store.search(query)
        if not relevant_chunks:
            return "No relevant information found."
        
        context = "\n\n".join([f"Document: {chunk.doc_name}\n{chunk.text}" 
                             for chunk in relevant_chunks])
        
        prompt = f"""**Context**:\n{context}\n**Question**: {query}\n
        Answer using ONLY the context. If unsure, say 'I don't know'."""
        return self.llm_service.generate_response(prompt)

class MultiAgentAssistant:
    """Main assistant coordinating tools and queries."""
    
    def __init__(self, vector_store: VectorStore, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.tools = [
            CalculatorTool(),
            DictionaryTool(),
            RAGTool(vector_store, llm_service)
        ]
        self.log = []
    
    def _select_tool(self, query: str) -> Tool:
        for tool in self.tools:
            if tool.name != "RAG" and tool.can_handle(query):
                return tool
        return next(tool for tool in self.tools if tool.name == "RAG")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        selected_tool = self._select_tool(query)
        response = selected_tool.execute(query)
        
        result = {
            "query": query,
            "tool_used": selected_tool.name,
            "response": response
        }
        
        if selected_tool.name == "RAG":
            retrieved_chunks = self.vector_store.search(query)
            result["retrieved_chunks"] = [
                {"document": chunk.doc_name, "chunk_id": chunk.chunk_id, "text": chunk.text}
                for chunk in retrieved_chunks
            ]
        
        self.log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **result
        })
        
        return result

def create_sample_documents() -> Dict[str, str]:
    """Create sample documents for demo purposes."""
    sample_docs = {
        "rag_overview.txt": """
        Retrieval-Augmented Generation (RAG) combines retrieval mechanisms with text generation capabilities of large language models.
        
        RAG works by first retrieving relevant documents or passages from a corpus based on a query, and then using these retrieved texts as additional context for the language model to generate a response.
        
        The key advantages of RAG include:
        1. Access to specialized knowledge not present in the LLM's training data
        2. Ability to cite sources and provide evidence for generated content
        3. Reduction in hallucinations (making up facts) by grounding responses in retrieved content
        4. More up-to-date information if the retrieval corpus is regularly updated
        
        The typical RAG pipeline involves:
        - Indexing: Converting documents into embeddings and storing them in a vector database
        - Retrieval: Finding the most relevant documents based on semantic similarity to the query
        - Generation: Using the retrieved documents as context for the language model to generate an answer
        
        Common challenges in RAG systems include ensuring retrieval quality, handling different types of queries appropriately, and balancing retrieved context with model knowledge.
        """,
        
        "llm_fundamentals.txt": """
        Large Language Models (LLMs) are a type of artificial intelligence model designed to understand and generate human language. They're trained on vast amounts of text data from the internet, books, articles, and other sources.
        
        LLMs work based on the Transformer architecture, which uses attention mechanisms to process text. During training, these models learn patterns in language by predicting missing words or next words in sequences.
        
        Key capabilities of modern LLMs include:
        - Text generation in various styles and formats
        - Question answering based on their training data
        - Summarization of long documents
        - Translation between languages
        - Code generation and understanding
        
        Popular LLMs include GPT models from OpenAI, Claude from Anthropic, LLaMA models from Meta, and various open-source alternatives.
        
        LLMs have limitations including potential for generating false information (hallucinations), reflecting biases present in training data, and an inability to access real-time information beyond their training cutoff.
        
        The field is rapidly evolving with improvements in model size, training techniques, and capabilities with each new generation of models.
        """,
        
        "vector_databases.txt": """
        Vector databases are specialized database systems designed to store and query vector embeddings efficiently. These embeddings are numerical representations of data (such as text, images, or audio) in a high-dimensional space.
        
        The core functionality of vector databases is similarity search - finding vectors that are close to a query vector according to distance metrics like cosine similarity or Euclidean distance.
        
        Popular vector database technologies include:
        
        - FAISS (Facebook AI Similarity Search): A library developed by Facebook Research for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, even ones that would not fit in RAM.
        
        - Pinecone: A fully-managed vector database service that makes it easy to build high-performance vector search applications. It handles the infrastructure and scaling automatically.
        
        - Weaviate: An open-source vector search engine that allows you to store objects and vector embeddings from various ML models and create semantic relations between them.
        
        - Chroma: A lightweight embedding database that makes it easy to build LLM apps with knowledge grounding.
        
        - Milvus: An open-source vector database built to power embedding similarity search and AI applications.
        
        Vector databases typically support features like:
        - Approximate Nearest Neighbor (ANN) search algorithms
        - Filtering capabilities to combine metadata and vector searches
        - Clustering and indexing methods to speed up retrieval
        - Horizontal scaling to handle large embedding collections
        
        They are fundamental components in modern AI applications, particularly in retrieval-augmented generation systems.
        """,
        
        "prompt_engineering.txt": """
        Prompt engineering is the practice of designing and optimizing inputs to language models to elicit desired outputs. It has become an essential skill for effectively working with large language models.
        
        Key techniques in prompt engineering include:
        
        1. Zero-shot prompting: Asking the model to perform a task without examples.
           Example: "Translate the following English text to French: 'Hello, how are you?'"
        
        2. Few-shot prompting: Providing the model with a few examples of the desired input-output behavior.
           Example: "English: Hello, French: Bonjour. English: Good morning, French: Bon matin. English: Thank you, French: ?"
        
        3. Chain-of-thought prompting: Asking the model to explain its reasoning step by step.
           Example: "What is 17 x 24? Let's solve this step by step."
        
        4. Role prompting: Giving the model a specific role or persona to assume.
           Example: "You are an expert mathematician specializing in calculus. Explain the concept of derivatives."
        
        5. Structured output prompting: Requesting responses in specific formats like JSON or CSV.
           Example: "List three capital cities and their countries in JSON format."
        
        Effective prompt engineering requires understanding the model's capabilities and limitations, being specific and clear in instructions, and iteratively refining prompts based on the model's responses.
        
        Common challenges include prompt injection attacks, handling ambiguous requests, and ensuring consistent output formats across different inputs.
        """,
        
        "multi_agent_systems.txt": """
        Multi-agent systems in AI refer to frameworks where multiple AI agents collaborate to solve complex problems. Each agent typically has specialized capabilities and knowledge, and they work together through coordination mechanisms.
        
        The key components of multi-agent systems include:
        
        1. Agent specialization: Different agents focus on different tasks or domains, such as:
           - Information retrieval agents
           - Calculation and reasoning agents
           - Planning and strategy agents
           - Code generation and execution agents
        
        2. Coordination frameworks: Mechanisms that enable agents to share information and delegate tasks, including:
           - Centralized controllers that dispatch tasks
           - Message-passing protocols between agents
           - Shared memory systems
           - Market-based mechanisms where agents bid for tasks
        
        3. Memory systems: Shared repositories that allow agents to store and access:
           - Task histories and outcomes
           - Relevant context from previous interactions
           - Intermediate results and reasoning steps
        
        Advantages of multi-agent approaches include:
        - Breaking down complex problems into manageable subproblems
        - Leveraging specialized capabilities for different aspects of a task
        - Improved robustness through redundancy and cross-validation
        - More transparent reasoning through explicit agent interactions
        
        Challenges include ensuring effective communication between agents, resolving conflicts when agents disagree, and managing the increased complexity of the overall system.
        
        Real-world applications include AutoGPT, BabyAGI, LangChain Agents, and systems where LLMs act as agents with access to tools and APIs.
        """
    }
    
    return sample_docs
def setup_streamlit_app():
    st.set_page_config(page_title="Multi-Agent Knowledge Assistant", layout="wide")
    st.title("Multi-Agent Knowledge Assistant")
    
    if 'assistant' not in st.session_state:
        vector_store = VectorStore()
        llm_service = LLMService(model_name="gemini-2.0-flash")
        document_processor = DocumentProcessor()
        
        sample_docs = create_sample_documents()
        all_chunks = []
        for doc_name, content in sample_docs.items():
            all_chunks.extend(document_processor.chunk_text(content, doc_name))
        
        vector_store.add_chunks(all_chunks)
        
        st.session_state.assistant = MultiAgentAssistant(vector_store, llm_service)
        st.session_state.document_processor = document_processor
        st.session_state.vector_store = vector_store
        st.session_state.query_history = []
    
    # Sidebar Document Management
    st.sidebar.header("Document Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload document (TXT/PDF/DOCX)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            content = st.session_state.document_processor.load_document(uploaded_file)
            chunks = st.session_state.document_processor.chunk_text(content, uploaded_file.name)
            st.session_state.vector_store.add_chunks(chunks)
            st.sidebar.success(f"Processed {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
    
    # Main Interface
    query = st.chat_input("Ask your question...")
    if query:
        with st.spinner("Processing..."):
            result = st.session_state.assistant.process_query(query)
            st.session_state.query_history.append(result)
        
        # Display Results
        with st.container():
            st.subheader("Answer")
            st.write(result["response"])
            
            if "retrieved_chunks" in result:
                with st.expander("Sources"):
                    for chunk in result["retrieved_chunks"]:
                        st.caption(f"**{chunk['document']} (Chunk {chunk['chunk_id']})**")
                        st.write(chunk["text"])
    
    # Query History
    if st.session_state.query_history:
        st.sidebar.subheader("Recent Queries")
        for entry in reversed(st.session_state.query_history[-5:]):
            st.sidebar.caption(f"{entry['query']} ({entry['tool_used']})")

if __name__ == "__main__":
    setup_streamlit_app()
