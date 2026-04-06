# LinguistFlow: End-to-End NMT Pipeline

[![CI Pipeline](https://github.com/walidght/NMT-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/walidght/NMT-pipeline/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/walidght/LinguistFlow-API)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)

An end-to-end Neural Machine Translation (NMT) project focused on fine-tuning a transformer model and deploying it as a production-ready microservice. This repository covers the full ML lifecycle: from data preprocessing and experiment tracking to containerized deployment.

**[View the Live API and Web UI on Hugging Face](https://huggingface.co/spaces/walidght/LinguistFlow-API)**

---

## The ML Lifecycle

### 1. Data & Model Training (The "Engine")
* **Fine-Tuning:** Leveraged Google Colab's GPU environment to fine-tune a `MarianMT` model (`Helsinki-NLP/opus-mt-en-fr`) on English-French parallel corpora.
* **Experiment Tracking:** Integrated **Weights & Biases (WandB)** to monitor training loss, validation metrics, and model checkpoints, ensuring reproducibility.
* **Preprocessing:** Implemented custom preprocessing scripts utilizing `SentencePiece` for tokenization and `sacremoses` for punctuation normalization to ensure data quality before training.

### 2. API & Deployment (The "Plumbing")
* **Service Layer:** Developed a FastAPI backend using the `lifespan` event to manage model loading, significantly reducing cold-start latency for inference.
* **Hybrid Interface:** Mounted a Gradio web UI at the root route (`/`) for manual testing while maintaining a high-throughput `/translate` POST endpoint for programmatic access.
* **CI/CD & DevOps:** Configured a GitHub Actions pipeline to run integration tests (`pytest` and `httpx`) on every push. Deployed automatically to Hugging Face Spaces via a `python:3.10-slim` Docker container to ensure environment parity.

---

## Local Development Setup

### 1. Clone & Install
```bash
git clone [https://github.com/walidght/NMT-pipeline.git](https://github.com/walidght/NMT-pipeline.git)
cd NMT-pipeline

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory. To avoid 429 Rate Limits from Hugging Face during model initialization, a token is required.
```env
LOCAL_MODEL_PATH=walidght/linguistflow-en-fr-mvp
HF_TOKEN=your_huggingface_read_token_here
```

### 3. Run the Server
```bash
uvicorn api.main:app --reload
```
* **Web UI:** `http://localhost:8000/`
* **API Docs (Swagger):** `http://localhost:8000/docs`

---

## Testing
The project uses `pytest` for automated integration testing.
```bash
pytest tests/ -v
```

---

## Docker Containerization
To ensure environment parity with the production cloud deployment, build and run the Docker container locally:

```bash
docker build -t linguistflow-api .
docker run -p 7860:7860 --env-file .env linguistflow-api
```