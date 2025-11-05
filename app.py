import os
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  


# LLM + embeddings + vector store
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain v1 imports
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage

# ------------- Config -------------
load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile" # Groq
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local + free

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("🔎📚 Askly")

# ------------- Session state -------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

def get_session_history(session_id: str):
    store = st.session_state.history_store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ------------- Helper function -------------
def format_docs(docs: list) -> str:
    return "\n\n".join([doc.page_content for doc in docs])

# ------------- Streaming callback -------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")
    def on_llm_end(self, response, **kwargs):
        self.container.markdown(self.text)

# ------------- Sidebar: Upload & Build KB -------------
st.sidebar.header("📂 Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Drop multiple PDFs / Word docs",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("💾 Build knowledge base"):
    with st.spinner("Embedding & indexing ..."):
        docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)

        for uf in uploaded_files:
            suffix = uf.name.split(".")[-1].lower()
            tmp_path = tempfile.mktemp(suffix="." + suffix)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())

            if suffix == "pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = Docx2txtLoader(tmp_path)

            docs.extend(loader.load_and_split(text_splitter=splitter))

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        persist_dir = f"db_{uuid.uuid4().hex}"
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()
        st.session_state.vector_db = vectordb
        st.toast("Knowledge base ready! Go chat 👉", icon="✅")

if st.session_state.vector_db is None:
    st.info("↖️ Upload & build your knowledge base first.")
    st.stop()

# ------------- Initialize LLM -------------
llm = ChatGroq(
    api_key=groq_api_key,
    model=LLM_MODEL,
    temperature=0.0,
    streaming=True
)

# ------------- Build retriever -------------
retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})

# ------------- Conversational Chain (fixed) -------------

def rag_chain(inputs):
    question = inputs["input"]
    chat_history = inputs["chat_history"]
    context = format_docs(retriever.invoke(question))
    system_message = SystemMessage(content=f"You are a helpful assistant. Use the context below.\n\nContext:\n{context}")
    messages = [system_message] + chat_history + [HumanMessage(content=question)]
    result = llm.invoke(messages)
    return {"answer": result.content}

conversational_chain = RunnableWithMessageHistory(
    RunnableLambda(rag_chain),
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ------------- Chat -------------
user_input = st.chat_input("Ask questions about your docs...")
if user_input:
    st.chat_message("user").markdown(user_input)

    answer_box = st.chat_message("assistant").empty()
    handler = StreamHandler(answer_box)

    result = conversational_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": st.session_state.session_id},
            "callbacks": [handler],
        },
    )
    answer_box.markdown(handler.text or result["answer"])

st.sidebar.divider()
st.sidebar.markdown(
    "Made with ❤️ using [Streamlit](https://streamlit.io) + [LangChain v1](https://python.langchain.com) + [Groq](https://console.groq.com)"
)