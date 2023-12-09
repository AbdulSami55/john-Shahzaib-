import json
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
from langchain.chains import RetrievalQA
from flask import Response
import time
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests


class RunpodServerlessLLM(LLM):
    pod_id: str
    api_key: str
    request_ids: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "runpod_serverless"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None and self._current_job_id is not None:
            #TODO: handle stop sequence
            ...
        response = self._run_generate_request(prompt)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"pod_id": self.pod_id}

    def _request_headers(self) -> Mapping[str, str]:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": self.api_key,
        }

    def _request_url(self) -> str:
        return f"https://api.runpod.ai/v2/{self.pod_id}"


    def _run_generate_request(self, prompt: str) -> str:
        headers = self._request_headers()
        input = {
            "method_name": "generate",
            "prompt": prompt,
            "sampling_params": {
           "max_tokens": 512,
            "tempreature":0.7
       }
            
        }

        
        # TODO: Handle network errors
        out = requests.post(
            f"{self._request_url()}/run",
            headers=headers,
            json={"input": input},
        ).json()

        id = out["id"]
        self.request_ids.append(id)

        while out["status"] != "COMPLETED":
            out = requests.get(
                f"{self._request_url()}/status/{id}",
                headers=headers,
            ).json()
            time.sleep(1)

        return out["output"]['text'][0]
    
llm = RunpodServerlessLLM(
    pod_id="uqfisnj4bkixcm",
    api_key="KXG7WNCE7Y41TNB4NI6JLZVVIHDFPBTIJ8UKJJFI",
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
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2})
)

    
 
def get_message(request):
    if request.method == 'GET':
        res = qa.run(request.args.get('message'))
        response =  Response(json.dumps(res), content_type='application/json')
        return response



