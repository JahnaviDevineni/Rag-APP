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
    # Keep your existing sample documents
    # ... (same as before)

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
