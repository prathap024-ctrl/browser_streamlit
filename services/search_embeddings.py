from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lib.llm_model import embeddings, llm
import chromadb
import os
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def search_embeddings(user_input):

    client = chromadb.CloudClient(
      api_key=os.getenv("CHROMA_API_KEY"),
      tenant=os.getenv("CHROMA_TENANT"),
      database=os.getenv("CHROMA_DATABASE")
    )
    
    vector_store = Chroma(
        client=client,
        collection_name="ai-browser",
        embedding_function=embeddings,
    )
    
    results = vector_store.similarity_search(
        user_input,
        k=2
    )

    for res in results:
        retrieved_text = f"*{res.page_content} [{res.metadata}]"
        
   #documents = results.get("documents", [])
   #if documents and documents[0]:
   #    retrieved_text = documents[0][0]
   #else:
   #     retrieved_text = ""

    search_prompt = PromptTemplate(
        input_variables=["text", "query"],
        template="""
    You are a helpful summarizer assistant.

    Given this content from a website:
    "{text}"

    And user query:
    "{query}"

    Write a concise, useful summary answering the query based on the content. Do not make up the answer, if you do not know the answer say "I don't know the answer!".
    """
    )

    summary_chain = search_prompt | llm | StrOutputParser()

    response = summary_chain.invoke({
        "text": retrieved_text,
        "query": user_input
    })

    return response

