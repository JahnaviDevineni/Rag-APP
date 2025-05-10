import os
import re
import math
import nltk
nltk.download("punkt_tab") 
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# For document processing and chunking
from nltk.tokenize import sent_tokenize
import PyPDF2
from docx import Document

# For embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss

import google.generativeai as genai

# For document processing and chunking
from nltk.tokenize import sent_tokenize
import PyPDF2
from docx import Document

# For embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss

# Download required NLTK data
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
print("NLTK data path:", nltk.data.path)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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
    
    def load_document(self, file_path: str) -> str:
        """Load document content based on file extension."""
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext.lower() == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif ext.lower() in ['.docx', '.doc']:
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
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
                # Create a chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(chunk_text, doc_name, chunk_id))
                chunk_id += 1
                
                # Handle overlap: remove sentences until we're below the overlap size
                overlap_size = 0
                while current_chunk and overlap_size < self.chunk_overlap:
                    overlap_size += len(current_chunk[0].split())
                    if overlap_size > self.chunk_overlap:
                        break
                    current_chunk.pop(0)
                
                current_size = sum(len(s.split()) for s in current_chunk)
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Don't forget the last chunk
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
        # Compute embeddings for all chunks
        for chunk in chunks:
            chunk.embedding = self.model.encode(chunk.text)
            self.chunks.append(chunk)
        
        # Create or update FAISS index
        embeddings = np.array([chunk.embedding for chunk in self.chunks]).astype('float32')
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        if len(self.chunks) > 0:
            # Clear the index and add all vectors
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 3) -> List[TextChunk]:
        """Find the most similar chunks to the query."""
        if not self.chunks or self.index is None:
            return []
        
        # Encode the query
        query_embedding = self.model.encode(query).reshape(1, -1).astype('float32')
        
        # Search for similar chunks
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        # Return the chunks
        results = [self.chunks[idx] for idx in indices[0]]
        return results

class Tool:
    """Base class for tools that agents can use."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def can_handle(self, query: str) -> bool:
        """Determine if this tool can handle the query."""
        raise NotImplementedError
    
    def execute(self, query: str) -> str:
        """Execute the tool on the query."""
        raise NotImplementedError

class CalculatorTool(Tool):
    """Tool for performing calculations."""
    
    def __init__(self):
        super().__init__(
            name="Calculator",
            description="Performs mathematical calculations"
        )
        # Define regex patterns for different calculations
        self.patterns = [
            r'(\d+\s*[\+\-\*\/\^]\s*\d+)',  # Basic operations with 2 numbers
            r'(what is|calculate|compute|solve|evaluate)\s+(.+)',  # Calculation commands
            r'(\d+\s*[\+\-\*\/\^][\d\s\+\-\*\/\^]*)',  # More complex expressions
        ]
    
    def can_handle(self, query: str) -> bool:
        """Check if the query is a calculation request."""
        query = query.lower()
        
        # Check for calculation keywords
        calc_keywords = ["calculate", "computation", "math", "sum", "add", "subtract", 
                         "multiply", "divide", "squared", "cubed", "power", "root", 
                         "percentage", "percent"]
        
        if any(keyword in query for keyword in calc_keywords):
            return True
        
        # Check for calculation patterns
        for pattern in self.patterns:
            if re.search(pattern, query):
                return True
        
        return False
    
    def execute(self, query: str) -> str:
        """Extract and evaluate the mathematical expression."""
        query = query.lower()
        expression = ""
        
        # Try to extract the mathematical expression
        for pattern in self.patterns:
            match = re.search(pattern, query)
            if match:
                if match.group(1) in ["what is", "calculate", "compute", "solve", "evaluate"]:
                    expression = match.group(2)
                else:
                    expression = match.group(1)
                break
        
        if not expression:
            expression = re.sub(r'[^0-9\+\-\*\/\^\(\)\.\s]', '', query)
        
        # Clean and prepare the expression
        expression = re.sub(r'x', '*', expression)  # Replace 'x' with '*'
        expression = re.sub(r'÷', '/', expression)  # Replace '÷' with '/'
        expression = re.sub(r'\^', '**', expression)  # Replace '^' with '**'
        
        try:
            # Safely evaluate the expression
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
        super().__init__(
            name="Dictionary",
            description="Looks up word definitions"
        )
        # Simple dictionary for demo purposes
        self.dictionary = {
            "rag": "Retrieval-Augmented Generation, a technique that enhances language models by retrieving relevant information from external knowledge sources.",
            "llm": "Large Language Model, a type of AI model trained on vast amounts of text data to generate human-like text.",
            "agent": "In AI, an autonomous entity that can perceive its environment, make decisions, and take actions to achieve goals.",
            "vector": "In machine learning, a mathematical representation of data as points in a multi-dimensional space.",
            "embedding": "A technique that maps discrete objects like words to vectors of real numbers in a continuous vector space.",
            "retrieval": "The process of finding and accessing relevant information from a data store.",
            "chunking": "The process of breaking down large texts into smaller, more manageable pieces.",
            "semantic": "Relating to meaning in language or logic.",
            "api": "Application Programming Interface, a set of rules and protocols for building and interacting with software applications.",
            "faiss": "Facebook AI Similarity Search, a library for efficient similarity search and clustering of dense vectors.",
            "prompt": "Input text that guides an AI model to generate relevant and contextual responses.",
            "streamlit": "An open-source Python library used to create web apps for machine learning and data science projects.",
            "python": "A high-level, interpreted programming language known for its readability and versatility.",
        }
    
    def can_handle(self, query: str) -> bool:
        """Check if the query is a definition request."""
        query = query.lower()
        
        # Check for definition keywords
        definition_keywords = ["define", "definition", "what is", "what are", 
                               "meaning of", "explain the term", "explain what"]
        
        if any(keyword in query for keyword in definition_keywords):
            # Extract potential word to define
            for word in self.dictionary:
                if word in query:
                    return True
        
        return False
    
    def execute(self, query: str) -> str:
        """Look up the definition of a word in the query."""
        query = query.lower()
        
        # Try to identify which word to define
        for word in self.dictionary:
            if word in query:
                return f"Dictionary definition for '{word}': {self.dictionary[word]}"
        
        # If no match found
        return "I couldn't find a definition for any word in your query in my dictionary."

class LLMService:
    """Service for interacting with Gemini models"""
    
    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

class RAGTool(Tool):
    def __init__(self, vector_store: VectorStore, llm_service):
        super().__init__(
            name="RAG",
            description="Retrieves information from documents using RAG"
        )
        self.vector_store = vector_store
        self.llm_service = llm_service
    
    def execute(self, query: str) -> str:
        """Retrieve relevant chunks and generate a response"""
        relevant_chunks = self.vector_store.search(query)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information in my knowledge base."
        
        context = "\n\n".join([f"Document: {chunk.doc_name}\n{chunk.text}" 
                             for chunk in relevant_chunks])
        
        prompt = f"""**Context**:\n{context}\n
**Question**: {query}\n
Answer the question using ONLY the provided context. 
If the answer isn't in the context, state that explicitly."""
        
        return self.llm_service.generate_response(prompt)


class MultiAgentAssistant:
    """Main assistant that coordinates tools and processes queries."""
    
    def __init__(self, vector_store: VectorStore, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.tools = []
        
        # Initialize tools
        self._initialize_tools()
        
        # Set up logging
        self.log = []
    
    def _initialize_tools(self):
        """Initialize all available tools."""
        self.tools.append(CalculatorTool())
        self.tools.append(DictionaryTool())
        self.tools.append(RAGTool(self.vector_store, self.llm_service))
    
    def _select_tool(self, query: str) -> Tool:
        """Select the appropriate tool for the query."""
        for tool in self.tools:
            if tool.name != "RAG" and tool.can_handle(query):
                return tool
        
        # Default to RAG if no specific tool matches
        return next(tool for tool in self.tools if tool.name == "RAG")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return a structured response."""
        # Select the appropriate tool
        selected_tool = self._select_tool(query)
        
        # Log the decision
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "tool_selected": selected_tool.name
        }
        
        # Execute the tool
        response = selected_tool.execute(query)
        log_entry["response"] = response
        
        # Add the log entry
        self.log.append(log_entry)
        
        # Prepare the result
        result = {
            "query": query,
            "tool_used": selected_tool.name,
            "response": response
        }
        
        # If RAG was used, include retrieved chunks
        if selected_tool.name == "RAG":
            retrieved_chunks = self.vector_store.search(query)
            result["retrieved_chunks"] = [
                {
                    "document": chunk.doc_name,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text
                }
                for chunk in retrieved_chunks
            ]
        
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
    """Set up the Streamlit web application."""
    st.title("Multi-Agent Knowledge Assistant")
    st.write("Ask questions about RAG, LLMs, and AI concepts, or try calculator and dictionary functions.")
    
    # Sidebar for document management
    st.sidebar.header("Document Management")
    
    # Initialize variables in session state
    if 'assistant' not in st.session_state:
        # Initialize vector store and LLM service
        vector_store = VectorStore()
        llm_service = LLMService(model_name="gemini-2.0-flash")
        
        # Initialize document processor
        document_processor = DocumentProcessor()
        
        # Create and process sample documents
        sample_docs = create_sample_documents()
        all_chunks = []
        
        for doc_name, content in sample_docs.items():
            chunks = document_processor.chunk_text(content, doc_name)
            all_chunks.extend(chunks)
        
        # Add chunks to vector store
        vector_store.add_chunks(all_chunks)
        
        # Initialize the assistant
        st.session_state.assistant = MultiAgentAssistant(vector_store, llm_service)
        st.session_state.document_processor = document_processor
        st.session_state.vector_store = vector_store
        st.session_state.query_history = []
    
    # Display loaded documents
    st.sidebar.subheader("Loaded Documents")
    doc_names = set(chunk.doc_name for chunk in st.session_state.vector_store.chunks)
    for doc_name in doc_names:
        st.sidebar.write(f"- {doc_name}")
    
    # Upload new documents
    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        # Save the uploaded file
        with open(f"uploaded_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the document
        content = st.session_state.document_processor.load_document(f"uploaded_{uploaded_file.name}")
        chunks = st.session_state.document_processor.chunk_text(content, uploaded_file.name)
        
        # Add to vector store
        st.session_state.vector_store.add_chunks(chunks)
        st.sidebar.success(f"Successfully added {uploaded_file.name}")
    
    # Main query interface
    query = st.text_input("Enter your question:")
    if st.button("Ask") and query:
        # Process the query
        with st.spinner("Processing..."):
            result = st.session_state.assistant.process_query(query)
            st.session_state.query_history.append(result)
        
        # Display the result
        st.subheader("Answer")
        st.write(result["response"])
        
        # Display the tool used
        st.subheader("Tool Used")
        st.write(result["tool_used"])
        
        # Display retrieved chunks if applicable
        if "retrieved_chunks" in result:
            st.subheader("Retrieved Information")
            for i, chunk in enumerate(result["retrieved_chunks"]):
                with st.expander(f"Document: {chunk['document']} (Chunk {chunk['chunk_id']})"):
                    st.write(chunk["text"])
    
    # Display query history
    if st.session_state.query_history:
        st.subheader("Query History")
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query: {item['query']}"):
                st.write(f"**Tool Used:** {item['tool_used']}")
                st.write(f"**Response:** {item['response']}")

if __name__ == "__main__":
    setup_streamlit_app()
