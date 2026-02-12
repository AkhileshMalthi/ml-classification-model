# ML Classification Model

A production-ready machine learning classification pipeline using scikit-learn, MLflow, FastAPI, and Docker. This project demonstrates model training, experiment tracking, model registry, and serving predictions via a REST API.

---

## Architecture

```
+-------------------+        +-------------------+        +-------------------+
|   Data Processor  |        |   Model Trainer   |        |   Inference API   |
| (Preprocessing)   | -----> | (MLflow Logging)  | -----> | (FastAPI + MLflow)|
+-------------------+        +-------------------+        +-------------------+
        |                           |                             |
        |                           |                             |
        +---------------------------+-----------------------------+
                                    |
                             +-------------------+
                             |   MLflow Server   |
                             | (Tracking & Artifacts)
                             +-------------------+
```

---

## Setup Instructions

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development)
- Git

### Clone the Repository

```sh
git clone https://github.com/AkhileshMalthi/ml-classification-model.git
cd ml-classification-model
```

---

## Running MLflow Experiments

1. **Install dependencies** (for local runs):

    ```sh
    pip install -r requirements.txt
    ```

2. **Train a model and log to MLflow:**

    ```sh
    python src/model_trainer.py
    ```

    You can specify dataset and hyperparameters:

    ```sh
    python src/model_trainer.py --dataset_name wine --C 0.5 --penalty l1
    ```

---

## Accessing the MLflow UI

After starting the MLflow server (see Docker instructions below), open:

```
http://localhost:5000
```

---

## Build and Run the Dockerized API

1. **Build and start all services:**

    ```sh
    docker-compose up --build
    ```

2. **Services:**
    - `mlflow_server`: MLflow tracking server (UI at port 5000)
    - `model_api`: FastAPI inference API (serving at port 8000)

---

## Testing the API

### Example: Health Check

```sh
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

### Example: Prediction

```sh
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Response:**
```json
{"prediction": [0]}
```

> Adjust the `features` array to match your model's expected input.

### Python Example

```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())
```

---

## Running Unit Tests

```sh
docker-compose exec model_api pytest
```

Or locally:

```sh
pytest tests/
```

---

## MLflow UI Screenshots

> Add screenshots here showing:
> - Experiment runs
> - Registered models
> - Metrics and artifacts (confusion matrix, scaler, etc.)

---

## Example Request/Response for `/predict`

**Request:**
```json
POST /predict
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response:**
```json
{
  "prediction": [0]
}
```

---

## Design Choices

- **MLflow** for experiment tracking, model registry, and artifact storage.
- **FastAPI** for high-performance, async REST API serving predictions.
- **Docker Compose** for reproducible, multi-service deployment.
- **Scikit-learn** for model training and preprocessing.
- **Joblib** for serializing the scaler as an MLflow artifact.
- **Unit tests** with pytest and FastAPI’s TestClient for API reliability.
- **Environment variables** for flexible configuration (MLflow URI, model name, etc.).
- **Lifespan event** in FastAPI to load model and scaler only once at startup.

---

## License

MIT License

---

**Happy experimenting!**