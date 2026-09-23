import io
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from app.inference import checkpoint_status, run_inference

app = FastAPI(title="DermaSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # Next.js dev server
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    primary_condition: str
    primary_confidence: float
    disease_family: str
    disease_subgroup: str
    hierarchy_path: list[str]
    hierarchy_confidence: dict[str, float]
    differentials: list[dict]
    hierarchical_consistency: bool | None
    explainability: dict
    model_version: str
    disclaimer: str


@app.get("/health")
def health():
    return {"status": "ok", "model": checkpoint_status()}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    age: int | None = Form(None),
    gender: str | None = Form(None),
    skin_tone: str | None = Form(None),
    body_location: str = Form(...),
    duration: str | None = Form(None),
    symptoms: str = Form(...),          # JSON string: '["Itching","Scaling"]'
):
    # Validate image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        parsed = json.loads(symptoms)
        if isinstance(parsed, list):
            symptoms_list = [str(x).strip() for x in parsed if str(x).strip()]
        elif isinstance(parsed, str):
            symptoms_list = [parsed.strip()] if parsed.strip() else []
        else:
            symptoms_list = []
    except Exception:
        symptoms_list = [s.strip() for s in symptoms.split(",") if s.strip()]

    # Read and decode image
    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Run model inference
    try:
        result = run_inference(
            pil_image=pil_img,
            age=age,
            gender=gender,
            skin_tone=skin_tone,
            body_location=body_location,
            duration=duration,
            symptoms=symptoms_list,
            top_k=3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return PredictResponse(**result)