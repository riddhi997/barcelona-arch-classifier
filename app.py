from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io

from model import build_model, CLASS_NAMES, THRESHOLDS, DESCRIPTIONS

app = FastAPI()

# load model once at startup
model = build_model()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze()

    max_prob, pred_idx = probs.max(0)
    pred_class = CLASS_NAMES[pred_idx.item()]
    confidence = round(max_prob.item(), 4)

    if confidence < THRESHOLDS[pred_class]:
        return JSONResponse({
            "prediction": "uncertain",
            "confidence": confidence,
            "top_class": pred_class,
            "description": f"The model could not classify this image with sufficient confidence. It most resembles {pred_class} but falls below the confidence threshold.",
            "all_probs": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASS_NAMES)}
        })

    return JSONResponse({
        "prediction": pred_class,
        "confidence": confidence,
        "description": DESCRIPTIONS[pred_class],
        "all_probs": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASS_NAMES)}
    })

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")