import os
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
)
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch


class AzureServices:
    """
    Class to encapsulate Azure Search and Azure OpenAI services configuration and functionality.
    """

    def __init__(self):
        # Load environment variables for Azure Search service
        self.azure_search_service_endpoint = os.environ.get('AZURE_SEARCH_SERVICE_ENDPOINT')
        self.azure_search_api_key = os.environ.get('AZURE_SEARCH_API_KEY')

        # Load environment variables for Azure OpenAI service
        self.azure_openai_chat_deployment_name = os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
        self.azure_openai_embeddings_deployment_name = os.environ.get('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME')
        self.azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION')
        self.azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
        self.azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')

        # Initialize the Azure Chat OpenAI model
        self.model = AzureChatOpenAI(
            deployment_name=self.azure_openai_chat_deployment_name,
            openai_api_version=self.azure_openai_api_version,
            openai_api_key=self.azure_openai_api_key,
            openai_api_base=self.azure_openai_endpoint,
            temperature=0,
            streaming=True
        )

        # Initialize the Azure OpenAI Embeddings model
        self.embeddings = OpenAIEmbeddings(
            deployment=self.azure_openai_embeddings_deployment_name,
            openai_api_version=self.azure_openai_api_version,
            openai_api_key=self.azure_openai_api_key,
            openai_api_base=self.azure_openai_endpoint,
        )

        # Define fields for uploaded files index
        self.uploaded_files_fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Adjust based on your embedding model dimensions
                vector_search_configuration="my-vector-config"
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SimpleField(
                name="thread_id",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True
            )
        ]

        # Initialize Azure Search vector store for user uploaded files
        self.uploaded_files_vector_store = AzureSearch(
            azure_search_endpoint=self.azure_search_service_endpoint,
            azure_search_key=self.azure_search_api_key,
            index_name="uploaded-files-index",
            embedding_function=self.embeddings.embed_query,
            fields=self.uploaded_files_fields,
            similarity="cosine"
        )

        # Define fields for RAG index
        self.rag_idx_fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Adjust based on your embedding model dimensions
                vector_search_configuration="my-vector-config"
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SimpleField(
                name="url",
                type=SearchFieldDataType.String,
                filterable=True
            )
        ]

        # Initialize Azure Search vector store for RAG (Retrieval-Augmented Generation)
        self.rag_vector_store = AzureSearch(
            azure_search_endpoint=self.azure_search_service_endpoint,
            azure_search_key=self.azure_search_api_key,
            index_name="rag-index",
            embedding_function=self.embeddings.embed_query,
            fields=self.rag_idx_fields,
            similarity="cosine"
        )