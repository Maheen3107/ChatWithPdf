from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import json
import os.path
from collections import defaultdict

# Load environment variables
load_dotenv()

# Set up Google API key for Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

def get_pdf_text(pdf_docs):
    text_with_pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.filename if hasattr(pdf, "filename") else "Unknown"
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                text_with_pages.append({
                    "text": page_text,
                    "page": i + 1,
                    "source": os.path.basename(pdf_name)
                })
    return text_with_pages

# Function to split text into chunks with page references
def get_text_chunks(text_with_pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    
    chunks_with_metadata = []
    
    for page_info in text_with_pages:
        page_chunks = text_splitter.split_text(page_info["text"])
        for chunk in page_chunks:
            chunks_with_metadata.append({
                "text": chunk,
                "page": page_info["page"],
                "source": page_info["source"]
            })
    
    # Store the metadata for later retrieval
    with open("chunk_metadata.json", "w") as f:
        json.dump(chunks_with_metadata, f)
        
    return chunks_with_metadata

# Function to generate embeddings and vector store
def get_vector_store(text_chunks_with_metadata):
    # Extract text for embeddings
    texts = [chunk["text"] for chunk in text_chunks_with_metadata]
    
    # Create metadata list
    metadatas = [{
        "page": chunk["page"],
        "source": chunk["source"]
    } for chunk in text_chunks_with_metadata]
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(
        texts, 
        embedding=embeddings, 
        metadatas=metadatas
    )
    vector_store.save_local("faiss_index")
    return "Vector store created successfully"

# Function to create a conversational chain for Q&A
def get_conversational_chain():
    """Create QA chain with page reference instructions."""
    prompt_template = """
    You are an expert PDF analyzer. Answer the question in detail using only the provided context.
    Always include the source document name and exact page number where the information was found.
    Format page references like this: [Source: filename.pdf, Page: X]
    
    If the answer isn't in the context, say: "I couldn't find this information in the documents."
    
    Context: {context}
    
    Question: {question}
    
    Detailed Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain


# Function to get search results from Google Custom Search API
def get_search_results(query, num_results=5):
    # Get API keys from environment variables
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    
    if not api_key or not cx:
        return {"error": "Google Search API key or CX not found in environment variables"}
    
    # Build the API URL
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": num_results
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            return {"error": f"Search API returned status code {response.status_code}"}
            
        search_results = response.json()
        
        if "items" not in search_results:
            return {"error": "No search results found"}
        
        # Extract relevant information from search results
        formatted_results = []
        for item in search_results["items"]:
            formatted_results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        
        return formatted_results
    except Exception as e:
        return {"error": f"Error performing search: {str(e)}"}

# Format search results as HTML
def format_search_results_html(search_results):
    if not search_results or isinstance(search_results, dict) and "error" in search_results:
        return "<p>No search results available.</p>"
    
    html = "<div class='search-results'><h3>Related Web Results:</h3><ul>"
    for result in search_results:
        html += f"<li><a href='{result['link']}' target='_blank'>{result['title']}</a>"
        html += f"<p>{result['snippet']}</p></li>"
    html += "</ul></div>"
    
    return html

# Enhance document display with page references
def format_docs_with_page_refs(docs):
    formatted_text = ""
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content
        formatted_text += f"\n[Source: {source}, Page: {page}]\n{content}\n"
    return formatted_text

# Improved function to format source citations
def format_sources_citation(docs):
    # Group pages by document
    sources_dict = defaultdict(set)
    
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown source")
        sources_dict[source].add(page)
    
    # Format the citation
    citations = []
    for source, pages in sources_dict.items():
        # Sort pages for readability
        sorted_pages = sorted(pages)
        # Format pages list nicely
        if len(sorted_pages) == 1:
            page_str = f"Page {sorted_pages[0]}"
        else:
            page_str = f"Pages {', '.join(str(p) for p in sorted_pages)}"
        
        citations.append(f"{source} ({page_str})")
    
    return "Sources: " + "; ".join(citations)

# WebSocket event for handling questions
@socketio.on('ask_question')
def handle_question(data):
    user_question = data.get('question')
    if not user_question:
        emit('response', {'message': 'No question provided.'})
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if FAISS index exists, if not, inform user
    if not os.path.exists("faiss_index"):
        emit('response', {'message': 'FAISS index not found. Please upload PDF and process it first.'})
        return

    # Load FAISS vector store
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        # Format docs with page references for context
        formatted_docs = format_docs_with_page_refs(docs)
        
        # Create conversational chain for question answering
        chain = get_conversational_chain()
        
        # Generate response based on the question and context (PDF chunks)
        response = chain(
            {
                "input_documents": docs, 
                "question": user_question
            }, 
            return_only_outputs=True
        )
        
        # Get a nicely formatted citation section
        citation_info = format_sources_citation(docs)
        
        # Add citation information
        answer_text = response.get("output_text", "Sorry, no response found.")
        full_answer = answer_text + "\n\n" + citation_info
        
        # Stream the main answer word by word
        for word in full_answer.split():
            emit('response', {'message': word + ' ', 'type': 'answer'})
            socketio.sleep(0.05)
        
        # Get Google search results
        search_results = get_search_results(user_question)
        
        # Format search results as HTML and send as a single message
        html_results = format_search_results_html(search_results)
        emit('response', {'message': html_results, 'type': 'search_results', 'format': 'html'})
    
    except Exception as e:
        emit('response', {'message': f"Error processing your question: {str(e)}", 'type': 'error'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'pdf_docs' not in request.files:
        return {'error': 'No file part'}, 400
    
    files = request.files.getlist('pdf_docs')
    if not files or files[0].filename == '':
        return {'error': 'No files selected'}, 400
    
    try:
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        result = get_vector_store(text_chunks)
        return {'message': 'Files processed successfully', 'detail': result}, 200
    except Exception as e:
        return {'error': f'Error processing files: {str(e)}'}, 500

@app.route('/status', methods=['GET'])
def check_status():
    has_index = os.path.exists("faiss_index")
    has_metadata = os.path.exists("chunk_metadata.json")
    
    return {
        'status': 'ready' if has_index and has_metadata else 'not_ready',
        'has_index': has_index,
        'has_metadata': has_metadata
    }

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)