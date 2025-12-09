# server.py
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
import torch.nn.functional as F
import torchvision.transforms as T

from film_unet import FiLM_UNet

# 🔹 메모리 로깅용
import psutil

# -----------------------------
# 기본 설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "film_unet_best.pth"  # ← 지금 폴더에 있는 weight 이름

logger = logging.getLogger("uvicorn.error")

# 전역 모델 핸들 (lazy load)
model = None  # type: ignore

# 입력 이미지 전처리
IMG_SIZE = 256   # 학습에 맞춰서 사용한 해상도로 맞추기
img_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),  # (C,H,W), 0~1
])


# -----------------------------
# 메모리 로깅 유틸
# -----------------------------
def log_memory(prefix: str = ""):
    """현재 프로세스의 RSS 메모리를 MB 단위로 로그에 출력"""
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[MEM] {prefix} {mem_mb:.1f} MB")
    except Exception as e:
        logger.error(f"[MEM] logging failed: {e}")


def get_model() -> FiLM_UNet:
    """
    FiLM_UNet 모델을 필요할 때 한 번만 로딩해서 전역으로 재사용.
    """
    global model

    if model is None:
        try:
            logger.info("🚀 FiLM_UNet 로드 시작")
            log_memory("before model load")

            # ★ 지금 올려준 모바일 U-Net 구조 그대로 사용
            m = FiLM_UNet(user_dim=4, base=16)

            # CPU로 먼저 로드
            state = torch.load(MODEL_PATH, map_location="cpu")
            m.load_state_dict(state)

            m.to(DEVICE)
            m.eval()

            model = m
            logger.info(f"✅ 모델 로드 완료: {MODEL_PATH} (device={DEVICE})")
            log_memory("after model load")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise

    return model


# -----------------------------
# 요청 바디 스키마
# -----------------------------
class CorrectionRequest(BaseModel):
    image: str              # base64 문자열 (jpg/png)
    user_vec: List[float]   # [protan, deutan, tritan, deltaE] 이런 식 4차원 벡터


# -----------------------------
# FastAPI 앱 + CORS
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 개발 단계: 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """
    서버 시작 시점에 한 번 모델 로드 + 메모리 로그.
    필요하면 lazy-load 하고 싶을 때는 get_model() 호출 부분을 주석 처리.
    """
    logger.info("🌐 서버 시작: startup 이벤트 호출")
    _ = get_model()


@app.get("/ping")
def ping():
    return {"message": "pong"}


# -----------------------------
# /correct 엔드포인트
# -----------------------------
@app.post("/correct")
def correct_color(req: CorrectionRequest):
    """
    입력:
      - image: base64 string (JPEG/PNG 등)
      - user_vec: [p, d, t, deltaE]  ← 4차원 유저 벡터

    출력:
      - {"corrected_image": "<base64 PNG>"}
    """
    try:
        # ---- user_vec 검증 ----
        if len(req.user_vec) != 4:
            raise HTTPException(
                status_code=400,
                detail=f"user_vec must be length 4, got {len(req.user_vec)}",
            )

        logger.info(f"📥 /correct called, user_vec={req.user_vec}")
        log_memory("before /correct")

        # ---- 모델 핸들 확보 ----
        m = get_model()

        # ---- 1) base64 → PIL 변환 ----
        try:
            img_bytes = base64.b64decode(req.image)
        except Exception as e:
            logger.error("Base64 decode error: %s", e)
            raise HTTPException(status_code=400, detail=f"base64 decode error: {e}")

        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            # 아이폰 세로사진 회전 보정 + RGB 변환
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            logger.error("PIL open/transpose error: %s", e)
            raise HTTPException(status_code=400, detail=f"PIL error: {e}")

        # ---- 2) 전처리 (IMG_SIZE x IMG_SIZE, Tensor) ----
        x = img_transform(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

        # 유저 벡터 텐서화 (1,4)
        user_vec = torch.tensor(
            [req.user_vec], dtype=torch.float32, device=DEVICE
        )

        # ---- 3) 모델 추론 ----
        with torch.no_grad():
            # FiLM_UNet forward(x, user_vec)
            y = m(x, user_vec)   # (1,3,H,W), 이미 sigmoid 통과 (0~1)

            # 필요하면 약간 smoothing
            y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

            # 원본과 블렌딩 (너무 과하게 안 바뀌게)
            alpha = 0.6  # 0.0 = 원본 / 1.0 = 모델 결과
            y = alpha * y + (1.0 - alpha) * x

        # ---- 4) 이미지 후처리 + base64 인코딩 ----
        y = y.squeeze(0).cpu().clamp(0, 1)  # (3,H,W)
        out_pil = T.ToPILImage()(y)

        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info("✅ /correct success")
        log_memory("after /correct")
        return {"corrected_image": out_b64}

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("❌ /correct unexpected error: %s\n%s", e, tb)
        log_memory("after /correct (error)")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# 로컬 테스트용 (Render에선 필요 X)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
