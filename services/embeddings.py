import chromadb
from lib.llm_model import embeddings
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

def embed_query(result, url):

    embedded_text = embeddings.embed_query(result)
    
    client = chromadb.CloudClient(
      api_key=os.getenv("CHROMA_API_KEY"),
      tenant=os.getenv("CHROMA_TENANT"),
      database=os.getenv("CHROMA_DATABASE")
    )
    
    collection=client.get_or_create_collection(name="ai-browser")
    
    doc_id = str(uuid.uuid4())
    collection.add(
        documents=[result],
        embeddings=[embedded_text],
        metadatas=[{"source": url["organic"][0]["link"]}],
        ids=[doc_id]
    )
    
    return embedded_text
