from fastapi import FastAPI
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceTextGenInference

app=FastAPI()

SERVER_URL='https://vq2gt1bsqyx09c-80.proxy.runpod.net'
llm = HuggingFaceTextGenInference(
    inference_server_url=SERVER_URL,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95
)

# Replace this with your Pinecone API key
API_KEY = "9b21c568-4634-4326-8ac9-f12af39545c8"
os.environ['OPENAI_API_KEY']='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'
embeddings = OpenAIEmbeddings()
pinecone.init(api_key=API_KEY,
              environment="gcp-starter")
              
index_name = 'test'

index = pinecone.Index(index_name)

docsearch = Pinecone(index,embeddings,'text')

print(docsearch.similarity_search('what is the budget of ride app',k=2))
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2})
)
    
@app.get('/message')  
def lambda_handler(message):
    res = qa.run(message)
    return {
        'statusCode': 200,
        'body': res
    }



