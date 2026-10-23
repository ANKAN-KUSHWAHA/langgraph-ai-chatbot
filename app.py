"""Streamlit interface for the Agentic LangGraph Chatbot."""
import os
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command

from chatbot import chatbot
from database import get_all_threads
from rag import ingest_rag_document


def generate_thread_id():
    # UUIDs remain internal identifiers; the sidebar displays readable titles.
    return str(uuid.uuid4())


def extract_text_content(content):
    """Return user-visible text only, ignoring Gemini metadata/signatures."""
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # Gemini/LangChain text content blocks can include extras/signatures.
        if content.get("type") == "text":
            text = content.get("text", "")
            return text if isinstance(text, str) else extract_text_content(text)

        # Some providers wrap content in another dictionary.
        for key in ("content", "text", "parts"):
            if key in content:
                extracted = extract_text_content(content[key])
                if extracted:
                    return extracted
        return ""

    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            extracted = extract_text_content(block)
            if extracted:
                parts.append(extracted)
        return "\\n".join(parts)

    # Unknown provider objects should not be stringified into the chat UI.
    return ""


def get_conversation_title(thread_id):
    """Use the first user message as a readable sidebar label."""
    try:
        messages = load_conversation(thread_id)
        for message in messages:
            if getattr(message, "type", None) == "human" or isinstance(message, HumanMessage):
                title = " ".join(extract_text_content(message.content).split())
                if title:
                    return title if len(title) <= 32 else title[:29].rstrip() + "..."
    except Exception:
        pass
    return "New conversation"


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_hitl"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def get_pending_interrupt(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = chatbot.get_state(config)
        direct_interrupts = getattr(snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]
        for task in (getattr(snapshot, "tasks", ()) or ()):
            interrupts = getattr(task, "interrupts", ()) or ()
            if interrupts:
                return interrupts[0]
    except Exception:
        return None
    return None


def save_pending_interrupt(thread_id, interrupt_object):
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id):
    pending = get_pending_interrupt(thread_id)
    if pending is not None:
        save_pending_interrupt(thread_id, pending)
    else:
        current = st.session_state.get("pending_hitl")
        if current and current.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    pending = st.session_state.get("pending_hitl")
    if not pending:
        st.warning("There is no pending action to approve or reject.")
        return

    thread_id = pending["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    try:
        with st.chat_message("assistant"):
            status = st.status("Resuming the requested action...", expanded=True)

            def stream_resumed_text():
                for chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, ToolMessage):
                        status.update(
                            label=f"Using tool: {getattr(chunk, 'name', 'tool')}",
                            state="running",
                            expanded=True,
                        )
                    elif isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                        text_content = extract_text_content(chunk.content)
                        if text_content:
                            yield text_content

            answer = st.write_stream(stream_resumed_text())
            next_interrupt = get_pending_interrupt(thread_id)
            if next_interrupt is not None:
                save_pending_interrupt(thread_id, next_interrupt)
                status.update(label="Another approval is required", state="complete")
            else:
                st.session_state["pending_hitl"] = None
                status.update(label="Action completed", state="complete")

        if answer:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": answer}
            )
        st.rerun()
    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")


st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("🤖Agentic Chatbot with LangGraph")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])
sync_pending_interrupt(st.session_state["thread_id"])

st.sidebar.title("My Conversations")
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    conversation_title = get_conversation_title(thread_id)
    if st.sidebar.button(conversation_title, key=thread_id, help=f"Conversation ID: {thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        history = []
        for message in messages:
            message_type = getattr(message, "type", None)
            if message_type == "human" or isinstance(message, HumanMessage):
                content = extract_text_content(message.content)
                if content:
                    history.append({"role": "user", "content": content})
            elif message_type == "ai" or isinstance(message, (AIMessage, AIMessageChunk)):
                # Show readable assistant text only, never provider metadata/signatures.
                content = extract_text_content(message.content)
                if content:
                    history.append({"role": "assistant", "content": content})
        st.session_state["message_history"] = history
        sync_pending_interrupt(thread_id)
        st.rerun()

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending = st.session_state.get("pending_hitl")
current_thread_has_pending_hitl = (
    pending is not None and pending.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    st.warning(f"Human approval required\n\n{pending['prompt']}")
    approve_col, reject_col = st.columns(2)
    with approve_col:
        if st.button("Approve Purchase", type="primary", use_container_width=True):
            resume_hitl_execution("yes")
    with reject_col:
        if st.button("Reject Purchase", use_container_width=True):
            resume_hitl_execution("no")

submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl,
)

if submission:
    user_input = (submission.text or "").strip()
    uploaded_files = submission.files or []

    if uploaded_files:
        pdf = uploaded_files[0]
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf.getvalue())
                temp_path = temp_file.name
            with st.spinner(f"Processing {pdf.name}..."):
                chunk_count = ingest_rag_document(temp_path)
            st.toast(f"{pdf.name} indexed successfully ({chunk_count} chunks).", icon="✅")
        except Exception as error:
            st.error(f"PDF processing failed: {error}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if user_input:
        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        with st.chat_message("assistant"):
            status_holder = {"box": None}

            def stream_answer():
                for chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, ToolMessage):
                        name = getattr(chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"Using tool: {name}", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"Using tool: {name}",
                                state="running",
                                expanded=True,
                            )
                    elif isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                        text_content = extract_text_content(chunk.content)
                        if text_content:
                            yield text_content

                interrupt_object = get_pending_interrupt(st.session_state["thread_id"])
                if interrupt_object is not None:
                    save_pending_interrupt(st.session_state["thread_id"], interrupt_object)
                    yield (
                        "\n\n⚠️ This simulated stock purchase needs your approval. "
                        "Use the Approve Purchase or Reject Purchase button below."
                    )

            answer = st.write_stream(stream_answer())
            if status_holder["box"] is not None:
                if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                    status_holder["box"].update(
                        label="Waiting for human approval",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status_holder["box"].update(
                        label="Tool finished",
                        state="complete",
                        expanded=False,
                    )

        st.session_state["message_history"].append(
            {"role": "assistant", "content": answer or ""}
        )
        if current_thread_has_pending_hitl or (
            st.session_state.get("pending_hitl")
            and st.session_state["pending_hitl"].get("thread_id")
            == st.session_state["thread_id"]
        ):
            st.rerun()
