from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from lib.llm_model import llm
from services.embeddings import embed_query

def search_and_summarize(search_query):
    search = GoogleSerperAPIWrapper()
    
    url_metadata = search.results(search_query)

    result=search.run(search_query)

    summary_prompt = PromptTemplate(
    input_variables=["text", "query"],
    template="""
        You are a helpful research assistant.

        Given this web content:
        "{text}"

        And user query:
        "{query}"

        Write a concise, useful summary answering the query based on the content. Use bullet points if relevant.
        """
        )
    
    summary_chain = summary_prompt | llm | StrOutputParser()

    response = summary_chain.invoke({
    "text": result,
    "query": search_query
    })
    embed_query(response,url_metadata)

    top_link = url_metadata.get("organic", [{}])[0].get("link", "No link found")

    return response, top_link
