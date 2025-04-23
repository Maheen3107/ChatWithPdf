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
from langchain.docstore.document import Document
from dotenv import load_dotenv
import re
from gtts import gTTS
import tempfile
import base64
from io import BytesIO
# Load environment variables
load_dotenv()

# Set up Google API key for Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# Function to extract text from PDF files with page tracking
def get_pdf_text(pdf_docs):
    """Extract text from PDF documents with page number metadata."""
    documents_with_metadata = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_name = os.path.basename(pdf.filename)
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    documents_with_metadata.append({
                        "text": page_text,
                        "metadata": {
                            "source": pdf_name,
                            "page": page_num
                        }
                    })
        except Exception as e:
            print(f"Error processing {pdf.filename}: {str(e)}")
    
    return documents_with_metadata

# Function to split text into chunks with metadata
def get_text_chunks(documents):
    """Split text into chunks while preserving page metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        text_chunks = text_splitter.split_text(doc["text"])
        for chunk in text_chunks:
            chunks.append(Document(
                page_content=chunk,
                metadata=doc["metadata"]
            ))
    
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    """Create and save FAISS vector store with metadata."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create conversational chain
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

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    """Convert text to speech and return as base64 encoded string."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return audio_data
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return None

# Add this new function for word-by-word streaming
def generate_word_stream(text):
    """Generator that yields text word by word with timing information."""
    words = text.split()
    for word in words:
        # Calculate delay based on word length (adjust as needed)
        delay = max(0.1, min(len(word) * 0.05, 0.5))  # Between 0.1 and 0.5 seconds
        yield {
            'word': word + ' ',
            'delay': delay
        }

# Modify the handle_question function
@socketio.on('ask_question')
def handle_question(data):
    """Handle questions with improved page reference tracking and TTS."""
    user_question = data.get('question')
    tts_enabled = data.get('tts', False)
    
    if not user_question:
        emit('response', {'message': 'Please ask a question.'})
        return
    
    try:
        # Load vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists("faiss_index"):
            emit('response', {'message': 'Please upload and process PDFs first.'})
            return
            
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question, k=4)
        
        # Collect source documents and their page numbers
        sources_with_pages = []
        for doc in docs:
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', 0)
                sources_with_pages.append(f"{source} (Page {page})")
        
        # Update prompt to encourage placing page references at the end
        prompt_template = """
        You are an expert PDF analyzer. Answer the question in detail using only the provided context.
        DO NOT include source citations or page numbers within your main answer text.
        
        At the very end of your response, list all sources used in this format: [Source: filename.pdf, Page: X]
        
        Context: {context}
        
        Question: {question}
        
        Detailed Answer:
        """
        
        # Generate response
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        # Extract the response text
        response_text = response.get("output_text", "Sorry, I couldn't generate a response.")
        
        # Remove any inline citations that might have been generated despite instructions
        response_text = re.sub(r'\[Source:.*?\]', '', response_text)
        
        # Add sources at the end if available, with proper formatting
        if sources_with_pages:
            # Remove any existing Sources section that might be in the response
            response_text = re.sub(r'\n\nSources:.*', '', response_text)
            
            # Add a clean Sources section
            unique_sources = list(set(sources_with_pages))
            response_text += "\n\nSources:"
            for source in unique_sources:
                response_text += f"\n- {source}"
        
        # Generate complete response for TTS
        if tts_enabled:
            # Clean text for TTS (remove citations, etc.)
            tts_text = re.sub(r'\[Source:.*?\]', '', response_text)
            tts_text = re.sub(r'\n\nSources:.*', '', tts_text)
            audio_data = text_to_speech(tts_text)
        else:
            audio_data = None
            
        # Emit the initial response with audio and full text
        emit('response_start', {
            'full_text': response_text, 
            'audio': audio_data
        })
        
        # Stream the response word by word
        for word_chunk in generate_word_stream(response_text):
            emit('response_chunk', word_chunk)
            socketio.sleep(word_chunk['delay'])
            
        emit('response_end', {'status': 'complete'})
            
    except Exception as e:
        emit('response', {'message': f"Error: {str(e)}"})

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle PDF upload and processing."""
    if 'pdf_docs' not in request.files:
        return {'error': 'No files uploaded'}, 400
        
    files = request.files.getlist('pdf_docs')
    if not files or files[0].filename == '':
        return {'error': 'No files selected'}, 400
    
    try:
        # Process PDFs
        documents = get_pdf_text(files)
        if not documents:
            return {'error': 'No text extracted from PDFs'}, 400
            
        # Create chunks with metadata
        text_chunks = get_text_chunks(documents)
        
        # Create and save vector store
        get_vector_store(text_chunks)
        
        return {'message': f'Processed {len(files)} PDF(s) successfully!'}, 200
        
    except Exception as e:
        return {'error': f'Processing failed: {str(e)}'}, 500

if __name__ == "__main__":
    socketio.run(app, debug=True)