# myapp/views.py

import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from django.shortcuts import render
from django.core.files.storage import default_storage
from .forms import PDFUploadForm
from sentence_transformers import SentenceTransformer
from .models import UploadedPDF
from django.core.files.storage import default_storage
from django.conf import settings
import string

# this model will be used for embedding
vector_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
index = None
text_chunks = []

def preprocess_text(text):
    """  Preprocess the input text by converting it to lowercase removing punctuation and tokenizing to words
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def extract_text_from_pdf(file_path):
    """ Extract text from a PDF file as store as string
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return preprocess_text(text)

def index_text_chunks(text):
    """Split the text into chunks and index them using FAISS.
    """
    global text_chunks, index
    text_chunks = text.split('\n\n') 
    embeddings = vector_model.encode(text_chunks) 
    index = faiss.IndexFlatL2(embeddings.shape[1]) 
    index.add(np.array(embeddings))  

def retrieve_relevant_chunks(query, index, model, text_chunks, top_k=5):
    """
    Retrieve the most relevant text chunks based on the query.
    """
    query_embedding = model.encode([query])  
    distances, indices = index.search(np.array(query_embedding), top_k)  
    relevant_chunks = [text_chunks[idx] for idx in indices[0]]  
    relevant_chunks_string = ', '.join(relevant_chunks) 
    return relevant_chunks_string

def generate_response(api_url, query, relevant_chunks):
    """
    Send a request to an GPT  API to generate a response based on the query.
    """
    payload = {
        "query": query,
        "text_input": relevant_chunks
    }
    try:
        response = requests.post(api_url, json=payload) 
        response.raise_for_status()  
        return response.json()  
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}  
    
def upload_pdf(request):
    """
    Handle PDF file upload, process the file, and return a response.
    """
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            query = form.cleaned_data['query']
            
            # Save the uploaded file to the media directory
            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            '''Extract text from the PDF'''
            text = extract_text_from_pdf(file_path)
            
            '''Index the text chunks'''
            index_text_chunks(text)
            
            '''Retrieve relevant chunks and generate a response'''
            relevant_chunks = retrieve_relevant_chunks(query, index, vector_model, text_chunks)
            api_url = "http://127.0.0.1:9000/generate-response/"
            response = generate_response(api_url, query, relevant_chunks)
            
            '''Render the result page with the text input and response'''
            return render(request, 'result.html', {
                'text_input': text,
                'response': response
            })
    else:
        form = PDFUploadForm()

    return render(request, 'upload.html', {'form': form})
