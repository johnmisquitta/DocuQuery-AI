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
