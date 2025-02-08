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

# ✅ 문서 임베딩 모델 로드
embedding_model = SentenceTransformer("all-mpnet-base-v2")

def create_database(target_folder):
    """FAISS 데이터베이스를 생성하는 함수"""
    if not target_folder or not os.path.exists(target_folder):
        st.error("❌ 올바른 폴더를 입력하세요.")
        return

    try:
        print(f"📂 대상 폴더: {target_folder}")

        # ✅ 문서 로드 (PDF, TXT, DOCX 지원)
        file_loader = FileLoader(target_folder, show_progress=True)
        documents = file_loader.load_all_files()
        print(f"🔍 로드된 문서 수: {len(documents)}")

        # ✅ 문서 전처리
        def preprocess_text(text):
            # 예시: 소문자 변환, 불용어 제거 등
            text = text.lower()  # 소문자 변환
            # 불용어 제거, 토큰화 등 추가 가능
            return text

        # ✅ 문서 분할 (긴 문서 지원) + 파일 경로 저장
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)

        docs = []
        metadata = []  # 파일 경로 저장
        original_texts = []  # 문서 원본 내용 저장

        for doc in documents:
            try:
                # 전처리 적용
                doc.page_content = preprocess_text(doc.page_content)
                split_docs = text_splitter.split_documents([doc])
                docs.extend(split_docs)
                metadata.extend([os.path.abspath(doc.metadata["source"])] * len(split_docs))  # 파일의 전체 경로 저장
                original_texts.extend([doc.page_content] * len(split_docs))  # 원본 내용 저장
            except Exception as e:
                print(f"⚠️ 파일 처리 중 에러 발생, 건너뜀: {doc.metadata['source']}, 에러: {e}")

        if not docs:
            print("❌ 문서를 찾을 수 없습니다.")
            sys.exit(1)

        # ✅ 문서 임베딩 모델 로드
        # embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        # embedding_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
        # embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # embedding_model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)
        # embedding_model = SentenceTransformer("jxm/cde-small-v2", trust_remote_code=True, device="cpu")
        embedding_model = SentenceTransformer("all-mpnet-base-v2")

        # ✅ 문서 벡터 변환
        texts = [doc.page_content for doc in docs]
        vectors = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"🔍 변환된 벡터 수: {len(vectors)}")

        # ✅ FAISS 인덱스 생성
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # ✅ FAISS 인덱스 저장
        
        if not os.path.exists("db"):
            os.makedirs("db")
        
        faiss.write_index(index, "db/faiss_index.idx")
        print("faiss checking point : ")
        # ✅ 문서 매핑 정보 저장 (파일 전체 경로 + 원본 내용 포함)
        with open("db/faiss_texts.pkl", "wb") as f:
            pickle.dump({"texts": texts, "metadata": metadata, "original_texts": original_texts}, f)

        st.info(f"🔄 데이터베이스를 생성 중... 📂 대상 폴더: {target_folder}")
        st.success("✅ 데이터베이스 생성 완료!")

    except Exception as e:
        st.error("❌ 데이터베이스 생성 실패!")
        st.text_area("에러 로그", str(e))
        print(f"에러 발생: {e}")

def search_faiss(query, top_k=5):
    """FAISS에서 검색하여 관련 문서를 반환"""
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
    """Ollama LLM을 사용하여 검색된 문서를 기반으로 답변 생성"""
    context = "\n\n".join([f"[{doc['file_name']}], {doc['summary']}" for doc in retrieved_docs])

    prompt = f"""
    사용자의 질문: {query}
    """
    
    system_prompt = f"""
    당신은 똑똑한 로컬 파일 탐색기 입니다. 질문을 바탕으로, 적절한 파일들을 추천해주고, 그 이유를 설명해 주어야 합니다. 파일에 대한 정보는 context 형태로 제공됩니다.
    다음은 관련 문서들에서 찾은 정보입니다:
    {context}
    대답은 반드시 한국어로 해야 합니다. 
    """

    # response = ollama.chat(model="deepseek-r1:8b", messages=[{"role": "user", "content": prompt}])
    response = ollama.chat(model="gemma2:2b", messages=[{"role": "system", "content" : system_prompt}, {"role": "user", "content": prompt}])
    return response["message"]["content"]

# 📌 Streamlit UI
st.title("📄 FAISS 기반 문서 검색 + LLM 답변")

# ✅ 사용자가 폴더 선택 후 "Create Database" 실행
target_folder = st.text_input("📂 대상 폴더 입력 (PDF, TXT, DOCX 지원):")

if st.button("🛠️ Create Database"):
    create_database(target_folder)

# ✅ 검색 UI
query = st.text_input("🔍 검색어를 입력하세요:")

if query:
    with st.spinner("검색 중..."):
        retrieved_docs = search_faiss(query)

        # 🔍 검색 결과 출력
        st.subheader("🔍 검색된 문서 목록:")
        for doc in retrieved_docs:
            st.markdown(f"**📄 파일명:** {doc['file_name']}")
            st.markdown(f"📂 **경로:** `{doc['full_path']}`")
            st.markdown(f"📜 **유사도:** {doc['score']}...\n")
            st.markdown(f"📂 **내용 요약:** `{doc['summary'][:300]}`")
            
            st.markdown("---")

        # 📝 LLM 응답 출력
        st.subheader("📝 LLM 답변:")
        llm_response = generate_llm_response(query, retrieved_docs)
        st.write(llm_response)
