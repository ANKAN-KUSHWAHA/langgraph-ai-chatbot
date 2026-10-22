# Agentic Chatbot with LangGraph

A Streamlit chatbot using Gemini, LangGraph, Tavily web search, a calculator,
weather and stock tools, simulated stock-purchase approval, and PDF RAG.

## Project structure

- `app.py` — Streamlit interface, conversation sidebar, streaming, and approval UI
- `chatbot.py` — LangGraph state, nodes, edges, and compiled agent
- `tools.py` — search, calculator, stock, weather, PDF retrieval, and approval tools
- `rag.py` — PDF ingestion and FAISS retrieval
- `database.py` — SQLite LangGraph checkpoint persistence and thread listing
- `config.py` — environment loading, local data paths, Gemini models
- `data/` — generated locally; contains SQLite and FAISS data

## Setup

1. Create and activate a virtual environment.
2. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your own API keys.
4. Run the app from this folder:

   ```bash
   streamlit run app.py
   ```

The SQLite database and FAISS index are stored in this project's `data/` folder,
so launching the app from another working directory will not accidentally use
a database relative to that directory. Do not copy the `data/` folder if you
want the new project to start with no previous conversations or PDF index.

## Important security notes

- Never commit `.env`, API keys, SQLite data, or FAISS indexes.
- If an API key was previously committed or exposed, revoke/rotate it. Removing
  it from the latest commit alone does not invalidate the key.
- FAISS loading uses dangerous deserialization for a local index. Only load
  indexes created by this app or otherwise trusted.
- `purchase_stock` is a simulation; it does not place a real trade.


## UI behavior

- Sidebar buttons display a shortened title derived from each conversation's first
  user message. UUIDs remain internal checkpoint IDs and appear only in the button
  tooltip.
- Gemini content blocks are filtered to their readable `text` field, so provider
  signatures/metadata are not printed as Python objects in the chat.
