import json
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from flask import Response
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter


API_KEY = "a4eee9dd-ba9d-41b3-b0a7-0cd6b069ff76"
os.environ['OPENAI_API_KEY']='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'
embeddings = OpenAIEmbeddings()
pinecone.init(api_key=API_KEY,
              environment="gcp-starter")
              
index_name = 'dataindex'

index = pinecone.Index(index_name)


def get_response(request):
    file = request.files["document"]
    if file:
        pdf = file.read()
        pdf_file = BytesIO(pdf)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page in range(num_pages):
            text += pdf_reader.pages[page].extract_text()

    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1500,
    chunk_overlap=150)

    record_texts = text_splitter.split_text(text)
    vectors=[]
    
    record_metadatas = [{
        "chunk": j, "text": text, "file_name":request.files["document"].filename
    } for j, text in enumerate(record_texts)]
    embeds = embeddings.embed_documents(record_texts)
    count=0
    for record_metadata in record_metadatas:
        vectors.append({'id':f"{record_metadata['chunk']}", 'values':embeds[count],'metadata':{'text':record_metadata['text'],'file_name':record_metadata['file_name']}})
        count+=1
    index.upsert(vectors)

    return Response(json.dumps({"status": "success"}), content_type='application/json')


