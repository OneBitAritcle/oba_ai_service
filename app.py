from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from bson.objectid import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from datetime import datetime
import os, re, json
from dotenv import load_dotenv

# ======================================
# 환경 변수 로드
# ======================================
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 누락!")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI 누락!")

# ======================================
# OpenAI 클라이언트
# ======================================
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================
# MongoDB 연결
# ======================================
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["OneBitArticle"]
collection = db["Selected_Articles"]

# ======================================
# FastAPI 설정
# ======================================
app = FastAPI(title="AI News Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ======================================
# Request Model
# ======================================
class AnalyzeRequest(BaseModel):
    article_id: str


# ======================================
# 🧩 GPT 처리 함수 (단일 기사 처리)
# ======================================
def generate_gpt_content(article_text: str):
    prompt = f"""
    다음 뉴스 내용을 기반으로 아래 요청을 수행해줘. 대상은 IT 직무 취업준비생이야.

    summary: 기사에서 전달하는 핵심 내용을 빠짐없이 포함하되, IT 취업준비생에게 특히 중요한 기술 동향·시장 변화·기업 전략 등을 중심으로 요약해줘.

    keywords: 기사 속에서 IT 취업준비생이 반드시 이해해야 하는 핵심 기술 개념 또는 최신 기술 트렌드를 10개 이내로 추출하고, 각 키워드는 신뢰할 수 있고 명확한 기술 설명을 붙여줘.

    quizzes: 기사를 읽고 학습한 내용을 점검할 수 있도록, 기사 내용 기반의 4지선다형 퀴즈 5개를 생성해줘. 각 퀴즈는 질문, 보기 4개, 정답 1개, 정답과 오답에 대한 상세한 해설을 포함해야 해.

    결과물은 반드시 아래 JSON 형식으로만 출력해야 하며, JSON 외의 추가 문장이나 설명은 절대 포함하면 안 돼.

    형식:
    {{
        "summary": "",
        "keywords": [
            {{"keyword": "", "description": ""}}
        ],
        "quizzes": [
            {{
                "question": "",
                "options": ["", "", "", ""],
                "answer": "",
                "explanation": ""
            }}
        ]
    }}

    뉴스 본문:
    \"\"\"{article_text[:7000]}\"\"\"  # 7k token 제한
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    raw_output = response.choices[0].message.content.strip()

    # JSON 추출
    json_match = re.search(r"\{[\s\S]*\}", raw_output)
    if not json_match:
        raise HTTPException(status_code=500, detail="GPT JSON 파싱 실패")

    return json.loads(json_match.group())


# 단일 GPT 처리 API (기존)
@app.post("/generate_gpt_result")
def generate_gpt_result(req: AnalyzeRequest):

    # ObjectId 검증
    try:
        oid = ObjectId(req.article_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="유효한 MongoDB ObjectId가 아닙니다.")

    # 문서 조회
    document = collection.find_one({"_id": oid})
    if not document:
        raise HTTPException(status_code=404, detail="해당 문서를 찾을 수 없습니다.")

    # content_col 합치기
    content_blocks = document.get("content_col", [])
    flat_lines = []

    for block in content_blocks:
        for line in block:
            if isinstance(line, str):
                flat_lines.append(line.strip())

    article_text = "\n".join(flat_lines)

    if not article_text:
        raise HTTPException(status_code=500, detail="content_col에서 본문을 추출할 수 없습니다.")

    # GPT 호출
    gpt_result = generate_gpt_content(article_text)

    # MongoDB 저장
    collection.update_one({"_id": oid}, {"$set": {"gpt_result": gpt_result}})

    return {
        "status": "OK",
        "message": "GPT 결과 저장 완료",
        "article_id": req.article_id,
        "gpt_result": gpt_result
    }


# 자동 처리 API: 오늘 날짜 기사 5개 자동 GPT 처리
@app.post("/generate_daily_gpt_results")
def generate_daily_gpt_results():
    today = datetime.now().strftime("%Y-%m-%d")

    # serving_date가 오늘인 문서 5개 찾기
    articles = list(collection.find({"serving_date": today}).limit(5))

    if not articles:
        return {"message": f"오늘 날짜({today}) 기사 없음"}

    updated_ids = []

    for article in articles:
        article_id = str(article["_id"])

        # 본문 추출
        content_blocks = article.get("content_col", [])
        flat_lines = []

        for block in content_blocks:
            for line in block:
                if isinstance(line, str):
                    flat_lines.append(line.strip())

        article_text = "\n".join(flat_lines)

        # GPT 호출
        gpt_result = generate_gpt_content(article_text)

        # 저장
        collection.update_one(
            {"_id": ObjectId(article_id)},
            {"$set": {"gpt_result": gpt_result}}
        )

        updated_ids.append(article_id)

    return {
        "status": "OK",
        "message": f"{len(updated_ids)}개 기사 GPT 자동 처리 완료",
        "processed_article_ids": updated_ids
    }
