import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

class DocxDirectoryLoader:
    """DOCX 파일을 디렉토리에서 로드하는 클래스"""
    def __init__(self, folder_path, show_progress=False):
        self.folder_path = folder_path
        self.show_progress = show_progress

    def load(self):
        documents : list[Document] = []
        count = 0
        print("🔍 DOCX 문서 로드 시작")
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.docx') or file.endswith('.doc'):
                    try:
                        docx_loader = Docx2txtLoader(os.path.join(root, file))
                        docx_documents = docx_loader.load()
                        documents.extend(docx_documents)
                        
                        count += 1
                        if self.show_progress:
                            print(f"✅ 로드된 파일: {file}")
                            # print(f"로드된 파일 내용 : {docx_documents}")
                            
                    except Exception as e:
                        print(f"⚠️ DOCX 파일 로드 중 에러 발생, 건너뜀: {file}, 에러: {e}")
        print(f"🔍 DOCX 문서 로드 완료: {count}개")
        return documents 