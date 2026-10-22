"""LangGraph workflow and compiled chatbot."""
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import llm
from database import checkpointer
from tools import TOOLS


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """You are a helpful Agentic Chatbot with access to several tools.

Tool usage instructions:
- Use rag_tool for questions about an uploaded PDF. Retrieve relevant document
  content before answering PDF-related questions.
- Use search_tool for current events, recent information, or information that
  requires an internet search.
- Use calculator for mathematical calculations.
- Use get_stock_price when asked for the current price of a stock.
- Use purchase_stock when the user asks to purchase a stock. This is a simulation
  and requires human approval.
- Use get_current_weather for current weather at a location.
Answer general questions directly when no tool is required. Do not invent facts
from uploaded documents. If no PDF is indexed, ask the user to upload one.
After receiving a tool result, provide a clear final answer.
"""

llm_with_tools = llm.bind_tools(TOOLS)


def chat_node(state: ChatState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

chatbot = builder.compile(checkpointer=checkpointer)
