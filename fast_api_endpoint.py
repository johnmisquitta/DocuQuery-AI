# import streamlit as st
# import os
# from datetime import datetime
# import fitz  # PyMuPDF
# import streamlit as st
# import io
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import openai
# model_name = 'paraphrase-MiniLM-L6-v2'
# vector_model = SentenceTransformer(model_name)
# index = None
# text_chunks = []

# def generate_response(context, query):
#     """
#     Generate a response using OpenAI's API based on the context and query.
#     """
#     prompt = f"Based on the following text, answer the query:\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
    
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=150
#     )
    
#     return response.choices[0].text.strip()
# # Function to save uploaded file to a specified directory with a new name
# def save_uploaded_file(uploaded_file, new_filename, save_directory):
#     try:
#         # Ensure the save directory exists
#         os.makedirs(save_directory, exist_ok=True)

#         # Define the full path to save the file
#         save_path = os.path.join(save_directory, new_filename)

#         # Save the file
#         with open(save_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         return save_path
#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None

# # Streamlit app
# def retrieve_relevant_chunks(query, index, model, text_chunks, top_k=5):
#     query_embedding = model.encode([query])
#     distances, indices = index.search(np.array(query_embedding), top_k)
#     relevant_chunks = [text_chunks[idx] for idx in indices[0]]
#     return relevant_chunks

# def index_text_chunks(text):
#     global text_chunks, index
#     text_chunks = text.split('\n\n')
#     embeddings = vector_model.encode(text_chunks)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings))

# def generate_response(query, relevant_chunks):
#     context = query + " ".join(relevant_chunks)
#     inputs = tokenizer.encode(context, return_tensors='pt')
#     outputs = gpt_model.generate(inputs, max_length=200)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response
# def main():
#     st.title("PDF Renamer and Saver")

#     # File uploader widget
#     uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

#     if uploaded_file is not None:
#         st.write("Uploaded file:", uploaded_file.name)

#         # Input widget for new filename

#         # Specify the save directory (you can change this as needed)
#         save_directory ="./pdf_files"

#         new_filename_with_extension =  "read.pdf"
#         if st.button("Save File"):
#             save_path = save_uploaded_file(uploaded_file, new_filename_with_extension, save_directory)
#             if save_path:
#                 st.success(f"File saved successfully to {save_path}")
#                 try:
#                     # Extract text from the uploaded file
#                     text = extract_text_from_pdf('./pdf_files/read.pdf')
#                     st.text_area("Extracted Text", text, height=400)
#                     st.success("Text extracted successfully!")
            
#                     with st.spinner("Indexing text chunks..."):
#                         index_text_chunks(text)
#                         st.success("Indexing completed!")
#                     query = "whats my name"
            
#                     with st.spinner("Retrieving relevant chunks..."):
#                         relevant_chunks = retrieve_relevant_chunks(query)
#                         st.write("Relevant Chunks:")
#                         st.write(relevant_chunks)
                        
#                     with st.spinner("Generating response..."):
#                         response = generate_response(query, relevant_chunks)
#                         st.success("Response generated!")
#                         st.write(response)


#                                         # Load pre-trained Sentence-BERT model
#                     # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#                     # # Segment the text (e.g., split into paragraphs)
#                     # text_chunks = text.split('\n\n')

#                     # # Vectorize the text chunks
#                     # embeddings = model.encode(text_chunks)

#                     # # Initialize Faiss index
#                     # index = faiss.IndexFlatL2(embeddings.shape[1])
#                     # index.add(np.array(embeddings))

#                     # # Save index for future use
#                     # faiss.write_index(index, 'document_index.faiss')
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")
#             else:
#                 st.error("Failed to save file")


# def extract_text_from_pdf(uploaded_file):
#     # Use io.BytesIO to read the uploaded file
#     doc = fitz.open(uploaded_file, filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text
# if __name__ == "__main__":
#     main()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

# Ensure you have your OpenAI API key set as an environment variable
# openai.api_key = os.environ.get("OPENAI_API_KEY")

class QueryRequest(BaseModel):
    query: str
    text_input: str

def generate_response(query: str, relevant_chunks: list[str]) -> str:
    context = " ".join(relevant_chunks)
    prompt = f"Based on the following text, answer the query:\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"

    try:
        client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-response/")
def post_response(request: QueryRequest):
    response = generate_response(request.query, request.text_input)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
