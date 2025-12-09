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
import torch.nn.functional as F

# ★ 모델 파일 가져오기
from film_unet import FiLM_UNet

# -----------------------------
# 기본 설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "film_unet_best.pth"

logger = logging.getLogger("uvicorn.error")
model = None 

# ★ 핵심 수정: 고정된 256 리사이즈 제거함 (동적으로 처리할 것임)
# MAX_SIZE: 너무 큰 이미지가 들어오면 서버 렉 걸리니까 이 정도로만 줄임 (화질 유지용)
MAX_SIZE = 1024 

# -----------------------------
# 모델 로드
# -----------------------------
def get_model() -> FiLM_UNet:
    global model
    if model is None:
        try:
            logger.info(f"🚀 모델 로드 시작 (Device: {DEVICE})")
            # 경량화 모델 (base=16)
            m = FiLM_UNet(user_dim=4, base=16)
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            model = m
            logger.info("✅ 모델 로드 성공!")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    return model

# -----------------------------
# 요청 데이터 구조
# -----------------------------
class CorrectionRequest(BaseModel):
    image: str              # base64
    user_vec: List[float]   # [p, d, t, delta]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    get_model()

# -----------------------------
# ★ 화질 살리는 핵심 함수
# -----------------------------
def smart_resize(img_tensor):
    """
    U-Net은 이미지 크기가 16의 배수여야 에러가 안 납니다. (Pooling 때문)
    이미지를 강제로 256으로 줄이는 대신, 가장 가까운 16의 배수로 살짝만 다듬습니다.
    """
    _, _, h, w = img_tensor.shape
    
    # 1. 너무 크면 줄이기 (메모리 보호)
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        _, _, h, w = img_tensor.shape # 줄어든 크기 업데이트

    # 2. 16의 배수로 맞추기 (Padding)
    # 예: 1000px -> 1008px (검은 테두리 살짝 추가해서 모델 오류 방지)
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    
    if pad_h > 0 or pad_w > 0:
        # (왼쪽, 오른쪽, 위, 아래) 순서로 패딩
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    return img_tensor, h, w  # 원본 높이/너비 반환 (나중에 잘라내기 위해)

# -----------------------------
# /correct 엔드포인트
# -----------------------------
@app.post("/correct")
def correct_color(req: CorrectionRequest):
    try:
        m = get_model()

        # 1. Base64 -> PIL -> Tensor
        try:
            img_bytes = base64.b64decode(req.image)
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Image decode fail")

        # 2. 텐서 변환 (0~1)
        # ★ 여기서 강제 리사이즈를 하지 않고 원본 그대로 텐서로 바꿉니다.
        x = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE) # (1, 3, H, W)

        # 3. ★ 스마트 리사이즈 (화질 보존의 핵심!)
        # 256으로 구겨넣지 않고, 원래 크기 근처에서 16배수만 맞춥니다.
        x_padded, orig_h, orig_w = smart_resize(x)
        
        user_vec = torch.tensor([req.user_vec], dtype=torch.float32, device=DEVICE)

        # 4. 모델 실행
        with torch.no_grad():
            y = m(x_padded, user_vec)

        # 5. 패딩 제거 (원래 크기로 복구)
        y = y[:, :, :orig_h, :orig_w]

        # 6. 결과 변환 및 전송
        y = y.squeeze(0).cpu().clamp(0, 1)
        out_pil = T.ToPILImage()(y)

        # 디버깅용 저장 (서버 폴더 확인해보세요 - 화질 좋아졌는지)
        out_pil.save("server_result_high_res.png")

        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=95) # 고화질 저장
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info(f"✅ 처리 완료: {orig_w}x{orig_h}")
        return {"corrected_image": out_b64}

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)