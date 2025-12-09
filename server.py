import io
import os
import base64
import logging
import traceback
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
# ★ 우리가 만든 경량화 모델 파일
from film_unet import FiLM_UNet
# :작은_파란색_다이아몬드: 메모리 로깅용
import psutil
# -----------------------------
# 기본 설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "film_unet_best.pth"  # 학습된 가중치 파일
logger = logging.getLogger("uvicorn.error")
# 전역 모델 핸들
model = None
# 입력 이미지 전처리
IMG_SIZE = 256
img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), # 강제로 256x256으로 맞춤 (잘림 방지)
    T.ToTensor(),  # (C,H,W), 0~1
])
# -----------------------------
# 메모리 로깅 유틸
# -----------------------------
def log_memory(prefix: str = ""):
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[MEM] {prefix} {mem_mb:.1f} MB")
    except Exception as e:
        logger.error(f"[MEM] logging failed: {e}")
def get_model() -> FiLM_UNet:
    global model
    if model is None:
        try:
            logger.info(":로켓: FiLM_UNet 로드 시작")
            log_memory("before model load")
            # ★ 경량화 모델 생성 (base=16 확인!)
            m = FiLM_UNet(user_dim=4, base=16)
            # 가중치 로드
            state = torch.load(MODEL_PATH, map_location="cpu")
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            model = m
            logger.info(f":흰색_확인_표시: 모델 로드 완료: {MODEL_PATH} (device={DEVICE})")
            log_memory("after model load")
        except Exception as e:
            logger.error(f":x: 모델 로드 실패: {e}")
            raise
    return model
# -----------------------------
# 요청 바디 스키마
# -----------------------------
class CorrectionRequest(BaseModel):
    image: str              # base64 문자열
    user_vec: List[float]   # [protan, deutan, tritan, deltaE]
# -----------------------------
# FastAPI 앱 + CORS
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def on_startup():
    logger.info(":자오선이_있는_지구: 서버 시작: startup 이벤트 호출")
    _ = get_model()
@app.get("/ping")
def ping():
    return {"message": "pong"}
# -----------------------------
# /correct 엔드포인트
# -----------------------------
@app.post("/correct")
def correct_color(req: CorrectionRequest):
    try:
        # ---- user_vec 검증 ----
        if len(req.user_vec) != 4:
            raise HTTPException(status_code=400, detail="user_vec length must be 4")
        logger.info(f":받은_편지함_트레이: /correct called, user_vec={req.user_vec}")
        # ---- 모델 가져오기 ----
        m = get_model()
        # ---- 1) base64 → PIL 변환 ----
        try:
            img_bytes = base64.b64decode(req.image)
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image decode error: {e}")
        # ---- 2) 전처리 (256x256 리사이즈) ----
        # 원본 크기 저장 (나중에 복원하고 싶다면 사용)
        original_size = pil_img.size
        x = img_transform(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,256,256)
        user_vec = torch.tensor([req.user_vec], dtype=torch.float32, device=DEVICE)
        # ---- 3) 모델 추론 (Inference) ----
        with torch.no_grad():
            # 모델이 뱉어낸 결과(y)가 곧 정답입니다.
            y = m(x, user_vec)
            # ★ 중요: 여기에 있던 avg_pool2d(블러)와 alpha blending(섞기)를 모두 제거했습니다.
            # 모델을 믿고 그대로 출력합니다.
        # ---- 4) 이미지 후처리 + base64 인코딩 ----
        y = y.squeeze(0).cpu().clamp(0, 1)  # (3, 256, 256)
        out_pil = T.ToPILImage()(y)
        # (선택사항) 원본 크기로 다시 키워서 보내고 싶다면 아래 주석 해제
        # out_pil = out_pil.resize(original_size, Image.BILINEAR)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        logger.info(":흰색_확인_표시: /correct success")
        return {"corrected_image": out_b64}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(":x: Error: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    # 외부 접속 허용 (0.0.0.0)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)