import os
import mimetypes
from datetime import datetime
from typing import Dict, Optional

import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.callbacks import AsyncLangchainCallbackHandler

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler

from dotenv import load_dotenv

from tools.rag_search import rag_search
from tools.web_search import web_search
from tools.uploaded_files_search import uploaded_files_search

from services.azure_services import AzureServices

# Load environment variables from .env file
load_dotenv()

# Add all supported mimetypes
mimetypes.add_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("application/pdf", ".pdf")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx")
mimetypes.add_type("text/plain", ".txt")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/bmp", ".bmp")
mimetypes.add_type("image/tiff", ".tiff")
mimetypes.add_type("image/heif", ".heif")
mimetypes.add_type("text/html", ".html")

azure_services = AzureServices()

DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_API_KEY = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")


# Callback handler for streaming responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = None

    async def on_llm_new_token(self, token: str, **kwargs):
        if not token:
            return
        if self.msg is None:
            self.msg = cl.Message(content="", author="Assistant")
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        if self.msg:
            await self.msg.send()
            self.msg = None


# Function to set up the agent
async def setup_agent(memory: ConversationSummaryBufferMemory):
    system_prompt = "You are a helpful assistant. The current date is " + datetime.now().strftime("%A, %Y-%m-%d")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent_tools = [rag_search, web_search]
    if cl.user_session.get("uploaded_files"):
        agent_tools.append(uploaded_files_search)

    agent = create_tool_calling_agent(
        azure_services.model, agent_tools, prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        memory=memory,
        max_iterations=5
    )

    cl.user_session.set("agent_executor", agent_executor)


# Chat start handler
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("current_thread", None)
    cl.user_session.set("uploaded_files", False)

    conversation_summary_memory = ConversationSummaryBufferMemory(
        llm=azure_services.model,
        max_token_limit=4000,
        memory_key="chat_history",
        return_messages=True
    )

    await setup_agent(conversation_summary_memory)


# Message handler
@cl.on_message
async def chat(message: cl.Message):
    cl.user_session.set("current_thread", message.thread_id)

    if message.elements:
        try:
            await file_loader(message)
        except Exception as e:
            await cl.Message(
                author="System",
                content="An error occurred while reading the file. Please try again.",
            ).send()

    agent_executor: AgentExecutor = cl.user_session.get("agent_executor")
    try:
        await agent_executor.ainvoke(
            {"input": message.content},
            {"callbacks": [AsyncLangchainCallbackHandler(), StreamHandler()]}
        )
    except Exception as e:
        await cl.Message(
            author="System",
            content="An error occurred while processing the message. Please try again.",
        ).send()


# Chat resume handler
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("current_thread", thread["id"])

    conversation_summary_memory = ConversationSummaryBufferMemory(
        llm=azure_services.model,
        max_token_limit=4000,
        memory_key="chat_history",
        return_messages=True
    )

    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            conversation_summary_memory.chat_memory.add_user_message(message["output"])
        else:
            conversation_summary_memory.chat_memory.add_ai_message(message["output"])

    await setup_agent(conversation_summary_memory)


# OAuth callback handler
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


# Chat end handler
@cl.on_chat_end
async def on_chat_end():
    cl.user_session.set("current_thread", None)