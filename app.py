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
            if page_text.strip():
                text_with_pages.append({
                    "text": page_text,
                    "page": i + 1,
                    "source": os.path.basename(pdf_name)
                })
    return text_with_pages

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
    with open("chunk_metadata.json", "w") as f:
        json.dump(chunks_with_metadata, f)
    return chunks_with_metadata

def get_vector_store(text_chunks_with_metadata):
    texts = [chunk["text"] for chunk in text_chunks_with_metadata]
    metadatas = [{
        "page": chunk["page"],
        "source": chunk["source"]
    } for chunk in text_chunks_with_metadata]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")
    return "Vector store created successfully"

def get_conversational_chain():
    prompt_template = """
    You are an expert PDF analyzer. Answer the user's question using only the provided context from the uploaded PDF documents.

    Format your response using these HTML tags for better readability:
    - Use <h2> for main headings
    - Use <h3> for subheadings
    - Use <b> for bold text
    - Use <ul> and <li> for bullet points
    - Use <mark> for highlighting important text
    - Use <br> for line breaks
    - Use <p> for paragraphs
    - Use <a> for links (if any)
    - Use <hr> for section separators

    If the answer is found in the context:
    1. Start with a clear <h2> heading
    2. Provide a <p> introduction paragraph
    3. Use <ul> and <li> for listing key points
    4. Use <mark> to highlight important findings
    5. Use <b> for emphasis on key terms
    6. Include source references at the end

    If the answer is not found in the context, respond with:
    "<p>This information was not found in the uploaded PDFs. However, here are some helpful web resources.</p>"

    Context: {context}

    Question: {question}

    Detailed Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_search_results(query, num_results=5):
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    if not api_key or not cx:
        return {"error": "Google Search API key or CX not found in environment variables"}
    url = "https://www.googleapis.com/customsearch/v1"
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

def get_image_search_results(query, num_results=5):
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    if not api_key or not cx:
        return {"error": "Google Search API key or CX not found in environment variables"}
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": num_results,
        "searchType": "image"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return {"error": f"Search API returned status code {response.status_code}"}
        search_results = response.json()
        if "items" not in search_results:
            return {"error": "No image results found"}
        formatted_results = []
        for item in search_results["items"]:
            formatted_results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "context": item.get("image", {}).get("contextLink", "")
            })
        return formatted_results
    except Exception as e:
        return {"error": f"Error performing image search: {str(e)}"}

def format_search_results_html(search_results):
    if not search_results or isinstance(search_results, dict) and "error" in search_results:
        return "<p>No search results available.</p>"
    html = "<div class='search-results'><h3>Related Web Results:</h3><ul>"
    for result in search_results:
        html += f"<li><a href='{result['link']}' target='_blank'>{result['title']}</a><p>{result['snippet']}</p></li>"
    html += "</ul></div>"
    return html

def format_image_results_html(image_results):
    if not image_results or isinstance(image_results, dict) and "error" in image_results:
        return "<p>No image results available.</p>"
    html = "<div class='image-results'><h3>Related Images:</h3><div class='image-grid'>"
    for result in image_results:
        html += f"""
            <div class='image-item'>
                <a href='{result['context']}' target='_blank'>
                    <img src='{result['thumbnail']}' alt='{result['title']}' loading='lazy'>
                    <div class='image-title'>{result['title']}</div>
                </a>
            </div>
        """
    html += "</div></div>"
    return html

def format_docs_with_page_refs(docs):
    formatted_text = ""
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content
        formatted_text += f"\n[Source: {source}, Page: {page}]\n{content}\n"
    return formatted_text

def format_sources_citation(docs):
    sources_dict = defaultdict(set)
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown source")
        sources_dict[source].add(page)
    citations = []
    for source, pages in sources_dict.items():
        sorted_pages = sorted(pages)
        page_str = f"Page {sorted_pages[0]}" if len(sorted_pages) == 1 else f"Pages {', '.join(str(p) for p in sorted_pages)}"
        citations.append(f"{source} ({page_str})")
    return "Sources: " + "; ".join(citations)

@socketio.on('ask_question')
def handle_question(data):
    user_question = data.get('question')
    if not user_question:
        emit('response', {'message': 'No question provided.'})
        return
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        emit('response', {'message': 'FAISS index not found. Please upload PDF and process it first.'})
        return
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        formatted_docs = format_docs_with_page_refs(docs)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer_text = response.get("output_text", "Sorry, no response found.")
        full_answer = answer_text
        if "not found in the uploaded PDFs" not in answer_text:
            citation_info = format_sources_citation(docs)
            full_answer += "\n\n" + citation_info
        for word in full_answer.split():
            emit('response', {'message': word + ' ', 'type': 'answer'})
            socketio.sleep(0.05)
        
        # Get and emit web search results
        search_results = get_search_results(user_question)
        html_results = format_search_results_html(search_results)
        emit('response', {'message': html_results, 'type': 'search_results', 'format': 'html'})
        
        # Get and emit image search results
        image_results = get_image_search_results(user_question)
        image_html = format_image_results_html(image_results)
        emit('response', {'message': image_html, 'type': 'image_results', 'format': 'html'})
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
