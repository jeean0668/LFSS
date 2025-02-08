# 프로젝트 이름

이 프로젝트는 [프로젝트의 목적이나 기능에 대한 간략한 설명]을(를) 제공합니다. 이 프로젝트는 [사용 기술이나 프레임워크]를 사용하여 개발되었습니다.

## 기능

- 기능 1: [기능에 대한 설명]
- 기능 2: [기능에 대한 설명]
- 기능 3: [기능에 대한 설명]

## 설치 방법

이 프로젝트를 로컬 환경에 설치하려면 다음 단계를 따르세요.

1. 저장소 클론

   ```bash
   git clone https://github.com/사용자명/프로젝트명.git
   ```

2. 디렉토리로 이동

   ```bash
   cd 프로젝트명
   ```

3. 의존성 설치

   ```bash
   npm install
   ```

   또는

   ```bash
   yarn install
   ```

4. 애플리케이션 실행

   ```bash
   npm start
   ```

   또는

   ```bash
   yarn start
   ```

## 사용 예시

프로젝트를 실행한 후, [사용 방법이나 예시]를 참고하여 사용하세요.

## 기여 방법

기여를 원하신다면, [기여 방법에 대한 설명]을 참고하세요.

## 라이선스

이 프로젝트는 [라이선스 이름] 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참고하세요.

## 프로젝트 설치 방법

```bash
pip install -r requirements.txt
```

## 프로젝트 실행 방법

```bash
streamlit run app.py
```

## 현재까지 문제 
- 2025.01.26 : 파일 하나만 업로드 가능, 여러개 업로드시 메모리 과열 발생.(14b 모델 사용) => 여러 파일 업로드시 오랜 시간 소요. VectorDB 적절 모델 선정 => weaviate, chroma, faiss 비교분석 (10개 pdf 파일 업로드 시 메모리 사용량 비교)

| DB 이름 | 메모리 사용량 | 속도 | 장점 | 단점 |
| --- | --- | --- | --- | --- |
| chroma | 0.86MB | 0.75s | 빠른 개발 | 대용량 데이터 처리 부담 |
| faiss | 0.86MB | 0.71s | 대용량 적합 | 초기 세팅 어려움 |
| weaviate | 31.2MB | 1.56s | 파일 실시간 업데이트 용이 | query 속도 느림 |

1000개 이상의 파일 업로드가 필요하고, 빠른 속도를 달성해야 하기 때문에 FAISS 사용 결정. 하지만 실시간 파일 업데이트 부분에서 문제가 있음. 

- 2025.01.26 : 파일 업로드 시 벡터 DB 초기화 및 저장 부분 추가. update_rag.py 파일에 vectorDB 초기화 및 저장 부분을, run.py 파일에 파일 업로드 부분을 추가. 

## How it works?

Before:
```bash
/home/user/sample_data/
├── 2025년_산재보험료율표_다운로드.pdf
├── [양식] 계좌입금신청서(사업자)양식.pdf
├── announcement.pdf
├── test.pdf
└── 로켓피치_AWS_우수상_아마존악어떼_김지안,홍수정,남혜원,김태수,오재혁,한대명_250123_143423.pdf

Question :
How to pay for the insurance?
```

Result:
```bash 
You have to visit the website and pay for the insurance. Web site is https://www.rocketpich.com/ you have to prepare your account information.

source :
1. [/home/user/sample_data/2025년_산재보험료율표_다운로드.pdf](https://www.rocketpich.com/wp-content/uploads/2025년_산재보험료율표_다운로드.pdf)

2. [/home/user/sample_data/계좌입금신청서(사업자)양식.pdf](https://www.rocketpich.com/wp-content/uploads/계좌입금신청서(사업자)양식.pdf)
```


export OPENAI_API_KEY=fake-key
export OPENAI_BASE_URL=http://localhost:11434/v1