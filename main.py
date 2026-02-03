
import time
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from htmlTemplate import css, user_template, bot_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain import HuggingFacePipeline
from sentence_transformers import SentenceTransformer


# ————————————————————————————————————————————————————————————————
# 1) Load Persian LLaMA model (Hugging Face)

def load_model():
    model_id = "Qwen/Qwen2-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True
    )
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        repetition_penalty=1.1,
        do_sample=True,
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=hf_pipe)


# ————————————————————————————————————————————————————————————————
# 2) Vectorization and Chain Logic

def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=200
    )
    return splitter.split_text(raw_text)


def get_vectorstore(text_chunks):
    embed = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )
    return faiss.from_texts(texts=text_chunks, embedding=embed)


def get_qa_chain(vector_store, llm):
    retriever = vector_store.as_retriever(
        #search_type="mmr",
        search_kwargs={"k": 6}
    )
    prompt = PromptTemplate(
        input_variables=['textbook_context', 'question', 'student_profile', 'previous_interactions'],
        template="""
You are a patient, engaging AI teaching assistant. Use ONLY the textbook context to answer.
If you can’t answer from the context, say "I don’t know."

Context:
{textbook_context}

History:
{previous_interactions}

Question:
{question}

Answer:
"""
    )
    return (
            {'textbook_context': retriever,
             'question': RunnablePassthrough(),
             'student_profile': RunnableLambda(lambda _: st.session_state.get('student_profile', {})),
             'previous_interactions': RunnableLambda(lambda _: st.session_state.get('conversation_history', []))
             }
            | prompt
            | llm
            | StrOutputParser()
    )

# ————————————————————————————————————————————————————————————————
# 4) Streamlit App
def main():
    st.set_page_config(page_title="Chatting with Your PDFs", page_icon="📚")
    st.markdown(css, unsafe_allow_html=True)

    pdf_docs = st.sidebar.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if st.sidebar.button("Process"):
        if not pdf_docs:
            st.sidebar.warning("Please upload PDFs first!")
        else:
            start = time.time()
            raw_text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text() or ""
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vectorstore(text_chunks)

            llm = load_model()

            conversation = get_qa_chain(vector_store, llm)
            retriever = vector_store.as_retriever(
                # search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
            )

            st.session_state.update({
                "conversation": conversation,
                "retriever": retriever,
                "llm": llm
            })
            st.sidebar.success(f"Processed in {time.time() - start:.1f}s")

    question = st.text_input("Ask a question from your docs")
    if question and st.session_state.get("conversation"):
        st.markdown(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        ans = st.session_state.conversation.invoke(question)
        st.markdown(bot_template.replace("{{MSG}}", ans), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

