from langchain.tools import tool
from pydantic import BaseModel, Field

from services.azure_services import AzureServices

# Initialize Azure services
azure_services = AzureServices()

class SearchInput(BaseModel):
    query: str = Field(
        description="Semantic search query."
    )

@tool("rag-search-tool", args_schema=SearchInput)
async def rag_search(query: str) -> list[dict]:
    """
    Perform a semantic search using Azure services.
    
    Args:
        query (str): The semantic search query.

    Returns:
        list[dict]: A list of search result dictionaries with keys 'page_content', 'url', and 'title'.
    """
    try:
        results = await azure_services.rag_vector_store.asimilarity_search(
            query=query, k=5
        )
        return [
            {
                "page_content": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "title": doc.metadata.get("title", "")
            }
            for doc in results
        ]
    except Exception as e:
        return [{"response": f"An error occurred during the search: {str(e)}"}]