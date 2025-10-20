from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import os, re, json
from dotenv import load_dotenv

# ✅ 환경 변수 로드 (.env에서 OPENAI_API_KEY 읽기)
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="AI News Quiz & Interview Content Generator")

# ✅ OpenAI 클라이언트 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY 환경변수가 없습니다. .env 파일을 확인하세요.")
client = OpenAI(api_key=api_key)


# ======================================
# ✅ 1. 요청 데이터 모델 정의
# ======================================
class UrlRequest(BaseModel):
    url: str


class NewsRequest(BaseModel):
    url: str


# ======================================
# ✅ 2. 공통: 뉴스 본문 크롤링 함수
# ======================================
def extract_article_text(url: str) -> str:
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # article 태그 안의 <p> 우선
        paragraphs = [p.get_text().strip() for p in soup.select("article p") if p.get_text().strip()]
        if not paragraphs:
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]

        article = "\n".join(paragraphs)
        if not article:
            raise ValueError("본문을 찾을 수 없습니다.")
        return article

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"크롤링 실패: {e}")


# ======================================
# ✅ 3. 뉴스 요약 + 퀴즈 생성 API (기존 기능)
# ======================================
@app.post("/analyze")
def analyze_article(request: UrlRequest):
    article = extract_article_text(request.url)

    prompt = f"""
다음 뉴스 기사를 바탕으로 JSON 형식으로 작성하세요.

{{
  "summary": "3~5문장 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "quizzes": [
    {{
      "question": "질문",
      "options": ["보기1", "보기2", "보기3", "보기4"],
      "answer": "정답 보기",
      "explanation": "해설"
    }}
  ]
}}

[기사 본문]
{article[:5000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        raw_output = response.choices[0].message.content.strip()
        match = re.search(r"\{[\s\S]*\}", raw_output)
        if not match:
            raise ValueError("모델 응답에서 JSON 구조를 찾을 수 없습니다.")

        parsed = json.loads(match.group())

        return {
            "summary": parsed.get("summary"),
            "keywords": parsed.get("keywords"),
            "quizzes": parsed.get("quizzes"),
            "result": None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 처리 실패: {e}")


# ======================================
# ✅ 4. 뉴스 기반 면접/학습 콘텐츠 생성 API (새 기능)
# ======================================
@app.post("/generate_news_content")
def generate_news_content(req: NewsRequest):
    """
    입력: 뉴스 기사 URL
    출력: 핵심 요약, 키워드, 면접형 질문, 객관식 퀴즈 포함 콘텐츠
    """
    try:
        # 📰 1. 기사 본문 크롤링
        res = requests.get(req.url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        article = "\n".join([p.get_text() for p in soup.select("article p")])

        if not article.strip():
            raise ValueError("⚠️ 기사 본문을 찾을 수 없습니다. HTML 구조를 확인하세요.")

        # 🧠 2. GPT 프롬프트 구성
        prompt = f"""
당신은 'AI 면접 대비용 뉴스 학습 콘텐츠 전문가'입니다.
아래 뉴스 기사를 기반으로 다음 항목을 작성하세요.

---
[1️⃣ 핵심 요약]
- 기사 내용을 3~5문장으로 요약 (산업 트렌드 중심)

[2️⃣ 주요 키워드 및 설명]
- 핵심 키워드 5개: 정의, 산업 내 의미, 면접 활용 포인트 포함

[3️⃣ PT형 면접 질문 5개]
- 사고력 중심 질문으로 작성, 각 질문 뒤에 [면접 포인트] 포함

[4️⃣ 객관식 퀴즈 5문항]
- 보기 4개, 정답, 해설 포함
---

[기사 본문]
\"\"\"{article[:5000]}\"\"\"
"""

        # 🤖 3. OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 뉴스 기반 학습 콘텐츠를 생성하는 전문가야. 명확하고 깔끔하게 구조화해줘."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        result_text = response.choices[0].message.content.strip()

        return {
            "url": req.url,
            "content": result_text,
            "result": "OK"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FastAPI 처리 실패: {e}")


# ======================================
# ✅ 5. CORS 설정
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
