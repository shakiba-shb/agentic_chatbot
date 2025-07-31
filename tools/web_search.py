import os
import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

# Fetch environment variables for Bing Search API
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")
BING_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT")

class SearchInput(BaseModel):
    query: str = Field(description="Web search query.")
    mkt: str = Field(default="en-US", description="Market code.")

@tool("web-search-tool", args_schema=SearchInput)
async def web_search(query: str, mkt: str = "en-US") -> list[dict]:
    """
    Perform a web search using the Bing Search API.

    Args:
        query (str): The search query.
        mkt (str): Market code for the search, default is 'en-US'.

    Returns:
        list[dict]: A list of search results containing title, URL, source, and snippet.
    """
    params = {'q': query, 'mkt': mkt, 'count': 5}
    headers = {'Ocp-Apim-Subscription-Key': BING_SEARCH_API_KEY}

    try:
        response = requests.get(BING_ENDPOINT, headers=headers, params=params)
        response.raise_for_status()

        search_results = response.json()
        result = search_results.get('webPages', {}).get('value', [])

        return [
            {
                'title': webpage.get('name'),
                'url': webpage.get('url'),
                'source': webpage.get('displayUrl'),
                'snippet': webpage.get('snippet')
            }
            for webpage in result
        ]
    except requests.RequestException as e:
        return [{"response": f"An error occurred during the search: {str(e)}"}]