import os
import subprocess
import streamlit as st
import faiss
import pickle
import ollama
from sentence_transformers import SentenceTransformer
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.loader.FileLoader import FileLoader

# ✅ Load document embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

def create_database(target_folder):
    """Function to create a FAISS database"""
    if not target_folder or not os.path.exists(target_folder):
        st.error("❌ Please enter a valid folder.")
        return

    try:
        print(f"📂 Target Folder: {target_folder}")

        # ✅ Load documents (supports PDF, TXT, DOCX)
        file_loader = FileLoader(target_folder, show_progress=True)
        documents = file_loader.load_all_files()
        print(f"🔍 Loaded Document Count: {len(documents)}")

        # ✅ Preprocess documents
        def preprocess_text(text):
            # Example: convert to lowercase, remove stopwords, etc.
            text = text.lower()  # Convert to lowercase
            # Additional steps like stopword removal, tokenization can be added
            return text

        # ✅ Split documents (supports long documents) + save file paths
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)

        docs = []
        metadata = []  # Save file paths
        original_texts = []  # Save original document content

        for doc in documents:
            try:
                # Apply preprocessing
                doc.page_content = preprocess_text(doc.page_content)
                split_docs = text_splitter.split_documents([doc])
                docs.extend(split_docs)
                metadata.extend([os.path.abspath(doc.metadata["source"])] * len(split_docs))  # Save full file paths
                original_texts.extend([doc.page_content] * len(split_docs))  # Save original content
            except Exception as e:
                print(f"⚠️ Error processing file, skipping: {doc.metadata['source']}, Error: {e}")

        if not docs:
            print("❌ No documents found.")
            sys.exit(1)

        # ✅ Load document embedding model
        # embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        # embedding_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
        # embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # embedding_model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)
        # embedding_model = SentenceTransformer("jxm/cde-small-v2", trust_remote_code=True, device="cpu")
        embedding_model = SentenceTransformer("all-mpnet-base-v2")

        # ✅ Convert documents to vectors
        texts = [doc.page_content for doc in docs]
        vectors = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"🔍 Number of converted vectors: {len(vectors)}")

        # ✅ Create FAISS index
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # ✅ Save FAISS index
        if not os.path.exists("db"):
            os.makedirs("db")
        
        faiss.write_index(index, "db/faiss_index.idx")
        # ✅ Save document mapping information (including full file paths and original content)
        with open("db/faiss_texts.pkl", "wb") as f:
            pickle.dump({"texts": texts, "metadata": metadata, "original_texts": original_texts}, f)

        st.info(f"🔄 Creating database... 📂 Target folder: {target_folder}")
        st.success("✅ Database creation complete!")

    except Exception as e:
        st.error("❌ Database creation failed!")
        st.text_area("Error log", str(e))
        print(f"Error occurred: {e}")

def search_faiss(query, top_k=5):
    """Search FAISS and return related documents"""
    index = faiss.read_index("db/faiss_index.idx")

    with open("db/faiss_texts.pkl", "rb") as f:
        data = pickle.load(f)
        metadata = data["metadata"]
        original_texts = data["original_texts"]

    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)

    results = sorted([
        {
            "file_name": os.path.basename(metadata[idx]),
            "full_path": metadata[idx], 
            "score": round(float(distances[0][i]), 4),
            "summary": original_texts[idx]
        }
        for i, idx in enumerate(indices[0])
        if idx != -1 and 0 <= idx < len(metadata)
    ], key=lambda x: x["score"], reverse=True)

    return results

def generate_llm_response(query, retrieved_docs):
    """Generate a response based on retrieved documents using Ollama LLM"""
    context = "\n\n".join([f"[{doc['file_name']}], {doc['summary']}" for doc in retrieved_docs])

    prompt = f"""
    User's question: {query}
    """
    
    system_prompt = f"""
    You are a smart local file explorer. Based on the question, you should recommend appropriate files and explain why. Information about the files is provided in the form of context.
    Here is the information found in the related documents:
    {context}
    The answer must be in Korean.
    """

    # response = ollama.chat(model="deepseek-r1:8b", messages=[{"role": "user", "content": prompt}])
    response = ollama.chat(model="gemma2:2b", messages=[{"role": "system", "content" : system_prompt}, {"role": "user", "content": prompt}])
    return response["message"]["content"]

# 📌 Streamlit UI
st.title("📄 Local File Semantic Search + LLM Response")

# ✅ Execute "Create Database" after user selects a folder
target_folder = st.text_input("📂 Enter target folder (supports PDF, TXT, DOCX):")

if st.button("🛠️ Create Database"):
    create_database(target_folder)

# ✅ Search UI
query = st.text_input("🔍 Enter search query:")

if query:
    with st.spinner("Searching..."):
        retrieved_docs = search_faiss(query)

        # 🔍 Display search results
        st.subheader("🔍 List of Retrieved Documents:")
        for doc in retrieved_docs:
            st.markdown(f"**📄 File Name:** {doc['file_name']}")
            st.markdown(f"📂 **Path:** `{doc['full_path']}`")
            st.markdown(f"📜 **Similarity:** {doc['score']}...\n")
            st.markdown(f"📂 **Content Summary:** `{doc['summary'][:300]}`")
            
            st.markdown("---")

        # 📝 Display LLM response
        st.subheader("📝 LLM Answer:")
        llm_response = generate_llm_response(query, retrieved_docs)
        st.write(llm_response)