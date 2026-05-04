# Barcelona Architecture Classifier

A convolutional neural network that classifies images of Barcelona buildings into four architectural styles, served via a FastAPI backend with a web UI.

## Styles

- **L'Eixample** — 19th century grid expansion, regular facades and uniform balconies
- **Gothic Quarter** — medieval Catalan Gothic, pointed arches and heavy stone
- **Modernista** — Catalan Art Nouveau (1888–1930), organic forms and ornate decoration
- **Olympic Modern** — contemporary architecture from the 1992 Olympic redevelopment

## Model

- Base: ResNet-18 pretrained on ImageNet
- Two-stage transfer learning: frozen backbone → partial fine-tuning of `layer4`
- Regularisation: Dropout (p=0.2), early stopping, learning rate 5e-5 in Stage B
- Per-class confidence thresholds derived from ROC curves to flag uncertain predictions
- ~300 images across 4 classes, ~87% validation accuracy

## Stack

- **PyTorch + torchvision** — model training and inference
- **FastAPI** — REST API with `/predict` endpoint
- **Docker** — containerised deployment
- **Vanilla HTML/CSS/JS** — frontend UI

## Run locally

**With Docker:**
```bash
docker build -t barcelona-classifier .
docker run -p 8000:8000 barcelona-classifier
```

**Without Docker:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

## API

**POST** `/predict`

Upload an image file and receive:

```json
{
  "prediction": "modernista",
  "confidence": 0.8821,
  "description": "Catalan Modernisme (roughly 1888–1930)...",
  "all_probs": {
    "eixample": 0.0412,
    "gothic_quarter": 0.0231,
    "modernista": 0.8821,
    "olympic_modern": 0.0536
  }
}
```

If confidence falls below the per-class threshold the prediction returns `"uncertain"` with the top candidate class.

## Project structure

```
├── app.py              # FastAPI app
├── model.py            # Model architecture, class names, thresholds, descriptions
├── model.pth           # Trained weights (Git LFS)
├── requirements.txt
├── Dockerfile
├── static/
│   └── index.html      # Web UI
├── utils/
│   └── dataset.py      # Data loading and transforms
└── notebook/
    └── cnn_barcelona.ipynb
```

## Assignment context

Built as part of an AI course assignment on transfer learning and CNN fine-tuning. The notebook covers the full pipeline: EDA, preprocessing, two-stage transfer learning, evaluation (confusion matrix, t-SNE feature visualisation, ROC curves), and error analysis.

Key finding: eixample and modernista are architecturally entangled — the Eixample district was built largely during the modernista era, making the class boundary inherently ambiguous. This is reflected in overlapping t-SNE clusters and confirmed as a data problem rather than a model problem.
