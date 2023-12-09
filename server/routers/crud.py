
import asyncio
from typing import AsyncIterable
from langchain.schema import HumanMessage,AIMessage
from langchain.callbacks import AsyncIteratorCallbackHandler
from .schemas import ChatHistory
from langchain.embeddings import OpenAIEmbeddings
import os
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory,ChatMessageHistory


API_KEY = "a4eee9dd-ba9d-41b3-b0a7-0cd6b069ff76"
start_key="sk-ykOlX7M"
end_key="ZmpNaOJ79EB"
os.environ['OPENAI_API_KEY']=f'{start_key}ZvQqFXZckMN54T3BlbkFJiTJ32jnPh{end_key}'
embeddings = OpenAIEmbeddings()
pinecone.init(api_key=API_KEY,
              environment="gcp-starter")
              
index_name = 'dataindex'

index = pinecone.Index(index_name)

def chatStreamingResponse(chat:ChatHistory,llm,docsearch):
    # try:
       
    chat_history=[]
    for data in chat.History:
        chat_history.append((data.User,data.AI))
    # memory=ConversationBufferMemory(chat_memory=chat_history)
    qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2})
    )
    res = qa.run({'question':chat.UserMessage,'chat_history':chat_history})
    return res


    # except Exception as e:
    #     print(f"Caught exception: {e}")







def add_embeddings(file):
    file = file.document
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
        "chunk": j, "text": text, "file_name":file.filename
    } for j, text in enumerate(record_texts)]
    embeds = embeddings.embed_documents(record_texts)
    count=0
    for record_metadata in record_metadatas:
        vectors.append({'id':f"{record_metadata['chunk']}", 'values':embeds[count],'metadata':{'text':record_metadata['text'],'file_name':record_metadata['file_name']}})
        count+=1
    index.upsert(vectors)

    return {"status": "success"}


