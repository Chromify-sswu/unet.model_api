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

# ★ 우리가 만든 모델 파일 가져오기
from film_unet import FiLM_UNet

# -----------------------------
# 기본 설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "film_unet_best.pth"

logger = logging.getLogger("uvicorn.error")
model = None

# ★ 핵심 설정: 화질 제한을 1024px로 넉넉하게 둠 (256 아님!)
MAX_SIZE = 1024  # 너무 큰 이미지는 서버가 터질 수 있어서 이 정도로 제한


# -----------------------------
# 모델 로드 함수
# -----------------------------
def get_model() -> FiLM_UNet:
    global model
    if model is None:
        try:
            logger.info(f"🚀 모델 로드 시작 (Device: {DEVICE})")

            # ★ 경량화 모델 생성 (base=16 확인!)
            m = FiLM_UNet(user_dim=4, base=16)

            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"{MODEL_PATH} not found")

            # 가중치 파일 로드
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
# 요청 데이터 구조 (Pydantic)
# -----------------------------
class CorrectionRequest(BaseModel):
    image: str              # base64 string (data:image/...;base64, ... 도 허용)
    user_vec: List[float]   # [protan, deutan, tritan, deltaE_or_severity]


# -----------------------------
# FastAPI 앱 설정
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
    get_model()  # 서버 켜질 때 모델 미리 로드


# -----------------------------
# user_vec 정규화 함수
# -----------------------------
def normalize_user_vec(raw_vec: List[float]) -> List[float]:
    """
    학습할 때 마지막 값은 0~1 severity(α)로 썼다고 가정.
    앱에서 0~100 같은 deltaE로 보냈을 수도 있으니 0~1로 스케일링해 줌.
    """
    if len(raw_vec) != 4:
        raise ValueError("user_vec length must be 4")

    p, d, t, s = raw_vec

    # s가 1보다 크면 deltaE 느낌이라고 보고 0~1로 축소
    if s > 1.0:
        s = max(0.0, min(s / 100.0, 1.0))
    else:
        s = max(0.0, min(s, 1.0))

    return [float(p), float(d), float(t), float(s)]


# -----------------------------
# ★ 화질 살리는 핵심 함수 (Smart Resize)
# -----------------------------
def smart_resize(img_tensor: torch.Tensor):
    """
    이미지를 256으로 구겨 넣지 않고, 원본 크기를 최대한 살립니다.
    단, U-Net이 작동하려면 가로/세로가 16의 배수여야 하므로
    이미지를 늘리지 않고 가장자리에 살짝 '여백(Padding)'을 줍니다.

    반환하는 (valid_h, valid_w)는
    - MAX_SIZE로 한 번 줄인 뒤
    - padding 넣기 직전의 '실제 유효 영역 크기' 입니다.
    """
    _, _, h, w = img_tensor.shape

    # 1. 이미지가 너무 크면(예: 3000px) 적당히(1024px) 줄여서 메모리 보호
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_tensor = F.interpolate(
            img_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        _, _, h, w = img_tensor.shape  # 줄어든 크기로 업데이트

    # 2. 16의 배수로 맞추기 (Padding)
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16

    if pad_h > 0 or pad_w > 0:
        # (왼쪽, 오른쪽, 위, 아래) 순서로 패딩 적용
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return img_tensor, h, w  # 패딩 붙이기 전의 '유효 영역 크기' 반환


# -----------------------------
# /correct 엔드포인트
# -----------------------------
@app.post("/correct")
def correct_color(req: CorrectionRequest):
    try:
        m = get_model()

        # 0. user_vec 정규화
        try:
            norm_vec = normalize_user_vec(req.user_vec)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"user_vec error: {e}")

        # 1. Base64 -> PIL -> Tensor 변환
        try:
            b64_str = req.image
            # data:image/png;base64,xxxx 형식이면 콤마 뒤만 사용
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image decode fail: {e}")

        # 2. 텐서 변환 (0~1 범위)
        # ★ 주의: 여기서 T.Resize(256)을 절대 쓰지 않습니다! 원본 그대로 변환.
        x = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

        # 3. ★ 스마트 리사이즈 적용
        x_padded, valid_h, valid_w = smart_resize(x)

        user_vec_tensor = torch.tensor(
            [norm_vec], dtype=torch.float32, device=DEVICE
        )

        # 4. 모델 실행 (Inference)
        with torch.no_grad():
            y = m(x_padded, user_vec_tensor)

        # 5. 패딩 제거 (resize 후 유효 영역만 깔끔하게 오려내기)
        y = y[:, :, :valid_h, :valid_w]

        # 6. 결과 변환 및 전송
        y = y.squeeze(0).cpu().clamp(0, 1)
        out_pil = T.ToPILImage()(y)

        # (디버깅용) 서버 컴퓨터 폴더에 결과 파일 저장
        # out_pil.save("server_result_check.png")

        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=95)  # 고화질 JPEG 저장
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info(
            f"✅ 처리 완료: {valid_w}x{valid_h} (UserVec: {norm_vec}, Device: {DEVICE})"
        )
        return {"corrected_image": out_b64}

    except HTTPException:
        # 이미 적절한 status 코드로 올린 에러는 그대로 통과
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in /correct: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # 0.0.0.0으로 설정하여 외부(앱)에서 접속 가능하게 함
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
