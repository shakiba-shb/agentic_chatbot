from langchain.tools import tool
from pydantic import BaseModel, Field
import chainlit as cl

from services.azure_services import AzureServices

# Initialize Azure services
azure_services = AzureServices()

class SearchInput(BaseModel):
    query: str = Field(description="Semantic search query.")

@tool("uploaded-files-search-tool", args_schema=SearchInput)
async def uploaded_files_search(query: str) -> list[dict]:
    """
    Perform a semantic search on uploaded files specific to the current thread.

    Args:
        query (str): The semantic search query.

    Returns:
        list[dict]: A list of search result dictionaries with keys 'page_content' and 'title'.
    """
    try:
        current_thread = cl.user_session.get("current_thread")
        filters = f"thread_id eq '{current_thread}'"
        results = await azure_services.uploaded_files_vector_store.asimilarity_search(
            query=query, k=5, filters=filters
        )

        return [
            {
                "page_content": doc.page_content,
                "title": doc.metadata.get("title", "")
            }
            for doc in results
        ]
    except Exception as e:
        return [{"response": f"An error occurred during the search: {str(e)}"}]