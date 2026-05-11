import os
from pathlib import Path
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration de la page Streamlit
st.set_page_config(page_title="Assistant Finance & Crypto", page_icon="📈", layout="centered")

DATA_DIR = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

def load_pdf_documents(data_dir: Path):
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    txt_paths = sorted(data_dir.glob("*.txt"))

    all_paths = pdf_paths + txt_paths

    if not all_paths:
        return []

    documents = []

    # Charger les PDFs
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)

    # Charger les TXT
    for txt_path in txt_paths:
        loader = TextLoader(str(txt_path), encoding="utf-8")
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = txt_path.name
        documents.extend(pages)

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def build_prompt():
    # CHANGEMENT ICI : Le rôle a été mis à jour pour la finance/crypto
    template = """
Tu es un assistant expert en finance, cryptomonnaies et blockchain.
Tu dois répondre uniquement à partir du contexte fourni.

Consignes importantes :
- Réponds en français.
- Si l’information n’est pas présente dans le contexte, dis clairement :
"Je ne trouve pas cette information dans les documents fournis."
- Ne donne pas de conseils financiers personnels, contente-toi d'analyser et de synthétiser les documents fournis.
- Donne une réponse claire, structurée et concise.
- Termine par une ligne "Sources :" avec les fichiers utilisés.

Contexte :
{context}

Question :
{question}
"""
    return PromptTemplate.from_template(template)

def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page = page + 1
        parts.append(f"[Extrait {i} | source={source} | page={page}]\n{doc.page_content}")
    return "\n\n".join(parts)

def format_sources(docs):
    unique_sources = []
    for doc in docs:
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page = page + 1
        item = f"{source} (page {page})"
        if item not in unique_sources:
            unique_sources.append(item)
    return ", ".join(unique_sources)

def answer_question(question, retriever, llm, prompt):
    docs = retriever.invoke(question)
    context = format_context(docs)
    sources = format_sources(docs)

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt).content

    if "Sources :" not in response:
        response = response.strip() + f"\n\nSources : {sources}"

    return response, docs


# GESTION STREAMLIT (Interface utilisateur)

@st.cache_resource(show_spinner=False)
def init_rag():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None, None, None, "La variable GROQ_API_KEY est absente du fichier .env."

    documents = load_pdf_documents(DATA_DIR)
    if not documents:
        return None, None, None, "Aucun fichier (.pdf ou .txt) trouvé dans le dossier 'data'."

    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_api_key)
    prompt = build_prompt()

    return retriever, llm, prompt, None

def main():
    
    st.title("📈 Assistant Finance & Crypto")
    
    with st.spinner("Analyse des documents financiers en cours..."):
        retriever, llm, prompt, error_msg = init_rag()

    if error_msg:
        st.error(error_msg)
        st.stop()

   
    if "messages" not in st.session_state:
        st.session_state.messages = [
            
            {"role": "assistant", "content": "Bonjour ! Posez-moi vos questions sur la finance, les cryptomonnaies ou la blockchain d'après les documents que vous avez fournis.", "docs": None}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Source de question
            if msg.get("docs"):
                with st.expander("Voir les extraits récupérés"):
                    for i, doc in enumerate(msg["docs"], start=1):
                        source = doc.metadata.get("source", "inconnue")
                        page = doc.metadata.get("page", "?")
                        if isinstance(page, int): page += 1
                        st.caption(f"**{i}. {source} | page {page}**")
                        st.text(doc.page_content[:300].replace("\n", " ") + "...")

    
    if user_question := st.chat_input("Ex: Que disent les documents sur le Bitcoin ou l'inflation ?"):
        
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans la base de données..."):
                answer, docs = answer_question(user_question, retriever, llm, prompt)
                
                st.markdown(answer)
                
                with st.expander("Voir les extraits récupérés"):
                    for i, doc in enumerate(docs, start=1):
                        source = doc.metadata.get("source", "inconnue")
                        page = doc.metadata.get("page", "?")
                        if isinstance(page, int): page += 1
                        st.caption(f"**{i}. {source} | page {page}**")
                        st.text(doc.page_content[:300].replace("\n", " ") + "...")
                        
        
        st.session_state.messages.append({"role": "assistant", "content": answer, "docs": docs})

if __name__ == "__main__":
    main()