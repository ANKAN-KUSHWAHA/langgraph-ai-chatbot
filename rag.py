"""PDF ingestion and retrieval-augmented generation helpers."""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import FAISS_PATH, embeddings


def ingest_rag_document(file_path: str) -> int:
    """Index a PDF into this project's local FAISS store. Returns chunk count."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The PDF did not contain any extractable text.")

    # Persist this project's document index under ./data/faiss_db.
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(FAISS_PATH))
    return len(chunks)


def get_retriever():
    """Load the local FAISS index; fail with a useful message if none exists."""
    if not (FAISS_PATH / "index.faiss").exists():
        raise FileNotFoundError(
            "No PDF index exists yet. Upload a PDF in the chat before asking "
            "questions about it."
        )

    vector_store = FAISS.load_local(
        folder_path=str(FAISS_PATH),
        embeddings=embeddings,
        # Only load indexes created by your own app. Do not load untrusted indexes.
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
