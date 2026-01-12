import streamlit as st
import os
import glob
import json
import torch
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS 
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Page Configuration ---
st.set_page_config(page_title="PC Build Assistant", page_icon="🖥️", layout="wide")

# --- Sidebar UI ---
with st.sidebar:
    st.title("Settings & Info")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    
    st.divider()
    st.info("System Status:")
    status_placeholder = st.empty()

# --- Resource Loading (Cached) ---
@st.cache_resource
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150, # Reduced to prevent token overflow
        temperature=0.1,    # Lower temperature = less hallucination/creative lying
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True     # Safety for long inputs
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    documents = []
    
    # JSON Data
    for filepath in glob.glob("*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for item in data:
                    content = ". ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in item.items() if v])
                    documents.append(Document(page_content=content, metadata={"source": filepath}))
        except Exception: pass

    # PDF Data
    pdf_file = "160 question of chatbot.pdf"
    if os.path.exists(pdf_file):
        try:
            loader = PyPDFLoader(pdf_file)
            data = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Smaller chunks
            documents.extend(splitter.split_documents(data))
        except Exception: pass

    if not documents:
        return None, None

    vectorstore = FAISS.from_documents(documents, embeddings)
    # k=2 to keep context short and avoid the "Input length" error
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 2})
    
    return llm, retriever

llm, retriever = initialize_system()

if llm and retriever:
    status_placeholder.success("✅ AI Engine Ready")
else:
    status_placeholder.error("❌ System Offline (No Data)")

# --- STRICT PROMPT TEMPLATE ---
# We explicitly tell the AI NOT to use its own knowledge.
custom_template = """<|system|>
You are a strict PC Hardware Assistant. 
RULES:
1. ONLY use the provided Context to answer.
2. If the answer is not in the Context, say: "I am sorry, I don't have information about that in my dataset."
3. Do NOT make up components or compatibility info.
4. Keep answers brief.

Context:
{context}</s>
<|user|>
Chat History: {chat_history}
Question: {question}</s>
<|assistant|>"""

prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=custom_template)

if "memory" not in st.session_state:
    # Added output_key to match ConversationalRetrievalChain
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        output_key="answer", 
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main Interface ---
st.title("🖥️ AI PC Build Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about your specific dataset components..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if llm and retriever:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=False
            )
            
            with st.spinner("Checking local dataset..."):
                try:
                    response = qa_chain.invoke({"question": user_input})
                    full_answer = response['answer']
                    
                    if "<|assistant|>" in full_answer:
                        full_answer = full_answer.split("<|assistant|>")[-1].strip()

                    st.markdown(full_answer)
                    
                    if response['source_documents']:
                        with st.expander("📚 Data Sources Used"):
                            for doc in response['source_documents']:
                                st.write(f"- {doc.metadata.get('source|', 'Local File')}")
                                
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})
                
                except ValueError as e:
                    st.error("The conversation is too long. Please click 'Clear Chat History' to reset.")