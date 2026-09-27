from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal, Annotated
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from sqlite_checkpoint import memory
import os
from langchain_core.messages import SystemMessage
from RAG_workflow import retriever
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite",
    temperature = 0,
    google_api_key = api_key
)

SYSTEM_PROMPT = """You are a helpful assistant with access to a tool called `retrieve_documents`, \
which searches a vector database built from a specific book (Deep Learning Book) and returns \
relevant passages.

Guidelines:
1. If the user's question is about concepts, explanations, or content that could be covered in \
the book (e.g. deep learning theory, architectures, training techniques, math foundations), \
ALWAYS call `retrieve_documents` first before answering — do not rely on your own memory for \
these questions, since the user expects answers grounded in this specific source.
2. If the user's question is general conversation (greetings, clarifications, meta-questions \
about the chat itself, or topics clearly unrelated to the book), answer directly without calling \
the tool.
3. After retrieving documents, base your answer primarily on the retrieved content. If the \
retrieved passages don't contain enough information to answer the question, say so clearly \
instead of making up an answer — you may supplement with general knowledge only if you \
explicitly tell the user you're doing so.
4. Do not fabricate citations, page numbers, or quotes. Only reference what's actually present \
in the retrieved chunks.
5. Keep answers concise and well-structured. Use bullet points or short paragraphs where it \
helps clarity, and avoid restating the entire retrieved chunk verbatim — synthesize it.
6. If a follow-up question depends on earlier conversation context, use the conversation history \
to understand intent, but still retrieve fresh documents if the question needs factual grounding \
you don't already have.
"""

@tool
def retrieve_documents(query : str)-> str:
    """The retrieve_documents function is used to retrieve relevant documents based on the user's query.
        It searches the vector database and returns the most relevant document chunks to help generate an accurate response."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

tools = [retrieve_documents]
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

class graphState(TypedDict):
    messages : Annotated[list,add_messages]

def chat_node(state : graphState) -> graphState:
    query = state['messages']
    if not any(isinstance(m,SystemMessage) for m in state['messages']):
        query = [SystemMessage(content=SYSTEM_PROMPT)] + query
    response = llm_with_tools.invoke(query)
    return {"messages":response}


graph = StateGraph(graphState)

graph.add_node("chatnode",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chatnode")
graph.add_conditional_edges("chatnode",tools_condition)
graph.add_edge("tools","chatnode")

workflow = graph.compile(checkpointer=memory)