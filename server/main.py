import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
from fastapi.responses import StreamingResponse
import uvicorn
from routers.schemas import ChatHistory
from routers.llm import RunpodServerlessLLM
from fastapi import FastAPI, File, UploadFile
from routers import crud

app = FastAPI() 

    
# llm = RunpodServerlessLLM(
#     pod_id="uqfisnj4bkixcm",
#     api_key="KXG7WNCE7Y41TNB4NI6JLZVVIHDFPBTIJ8UKJJFI",
# )

from langchain.llms import HuggingFaceTextGenInference

SERVER_URL="https://qfle7l1jguvllr-80.proxy.runpod.net"
llm = HuggingFaceTextGenInference(
    inference_server_url=SERVER_URL,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95
)

# Replace this with your Pinecone API key
API_KEY = "a4eee9dd-ba9d-41b3-b0a7-0cd6b069ff76"
start_key="sk-ykOlX7M"
end_key="ZmpNaOJ79EB"
os.environ['OPENAI_API_KEY']=f'{start_key}ZvQqFXZckMN54T3BlbkFJiTJ32jnPh{end_key}'
embeddings = OpenAIEmbeddings()
pinecone.init(api_key=API_KEY,
              environment="gcp-starter")
              
index_name = 'dataindex'
index = pinecone.Index(index_name)
docsearch = Pinecone(index,embeddings,'text')

 

@app.post("/chat")
async def get_message(chat:ChatHistory):
    response = crud.chatStreamingResponse(chat,llm,docsearch=docsearch)
    return StreamingResponse(response, media_type='text/event-stream')
    
@app.post("/upload-docs")
async def upload_document(file:UploadFile=File(...)):
    return crud.add_embeddings(file)

if __name__=="__main__":
    uvicorn.run(app,host="192.168.18.84",port=8000)



