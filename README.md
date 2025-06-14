# YOLOv8 Segmentation API

A production-ready FastAPI service wrapping a YOLOv8 segmentation model. Provides an HTTP endpoint to submit images and receive detected object outlines (polygons), classes, and confidence scores. Designed for easy integration into Flutter, mobile apps, or any client.

---

## 🚀 Features

- **Instance Segmentation** using Ultralytics YOLOv8-Seg
- Returns **polygon outlines** for each detected object
- Robust **error handling** & **retries**
- **OOP**-style modular codebase
- Auto-reload in development, Uvicorn + Gunicorn for production
- Dockerized & deployable on Fly.io (or any container host)
- Configurable via environment variables
- **Logging** + metrics ready (via Python `logging`)

---

## 📦 Repository Structure

```
├── api/
│   ├── __init__.py
│   ├── config.py          # Env var loader
│   ├── errors.py          # Custom exception classes
│   ├── model.py           # YOLO wrapper & retry logic
│   ├── schemas.py         # Pydantic request/response models
│   ├── server.py          # FastAPI app & routers
│   └── utils.py           # Helpers (image I/O, contour → polygon)
├── Dockerfile             # Multi-stage build
├── fly.toml               # Fly.io config
├── requirements.txt       # Pin dependencies
└── README.md
```

---

## ⚙️ Configuration

| Env var                  | Default         | Description                                          |
|--------------------------|-----------------|------------------------------------------------------|
| `MODEL_PATH`             | `best.pt`       | Local path or S3/URL of YOLO `.pt` file              |
| `LOG_LEVEL`              | `INFO`          | Python logging level (DEBUG, INFO, WARNING, ERROR)   |
| `MAX_RETRIES`            | `3`             | Model-inference retry attempts on failure            |
| `RETRY_BACKOFF_SECONDS`  | `1`             | Backoff delay between retries                        |
| `HOST`                   | `0.0.0.0`       | Uvicorn bind host                                    |
| `PORT`                   | `7860`          | Uvicorn bind port                                    |

You can set these in a `.env` or as secrets in Fly.io.

---

## 🛠️ Installation & Development

1. **Clone & enter**  
   ```bash
   git clone https://github.com/D0N4C3/tools-instance-segmentation.git
   cd tools-instance-segmentation
   ```

2. **Python venv**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install deps**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run local server**  
   ```bash
   uvicorn api.server:app --reload --host ${HOST:-127.0.0.1} --port ${PORT:-8000}
   ```

---

## 🔌 API Endpoints

### `POST /segment`

- **Description:** Run instance segmentation on an image
- **Consumes:** `multipart/form-data` with key `file` (image/jpeg, image/png)
- **Produces:** `application/json`

#### Request

```http
POST /segment HTTP/1.1
Content-Type: multipart/form-data; boundary=---XYZ

-----XYZ
Content-Disposition: form-data; name="file"; filename="image.jpg"
Content-Type: image/jpeg

<...binary image bytes...>
-----XYZ--
```

#### Response (200 OK)

```json
[
  {
    "class": "Hammer",
    "confidence": 0.97,
    "polygon": [[12,34],[15,37],...]
  },
  {
    "class": "Drill",
    "confidence": 0.82,
    "polygon": [[100,200],[105,205],...]
  }
]
```

#### Error Responses

- **400 Bad Request**  
  ```json
  { "detail": "No file part in request" }
  ```
- **422 Unprocessable Entity** (invalid image)  
- **500 Internal Server Error**  
  ```json
  { "detail": "Segmentation failed: <error message>" }
  ```

---

## 🔄 Retry & Timeout

- On inference failures, the model wrapper will retry up to `MAX_RETRIES` times with exponential backoff.
- Overall request timeout is configurable in the HTTP client.

---

## 📦 Docker

Build & run locally:

```bash
docker build -t tools-instance-segmentation:latest .
docker run -e MODEL_PATH=best.pt -p 7860:7860 tools-instance-segmentation:latest
```

---

## ☁️ Deploy to Fly.io

1. **Install Fly CLI**: https://fly.io/docs/hands-on/install-flyctl/  
2. **Login**:
   ```bash
   fly auth login
   ```
3. **Launch** (first time only):
   ```bash
   fly launch --image tools-instance-segmentation:latest
   ```
4. **Deploy** (after code changes):
   ```bash
   fly deploy
   ```
5. **Logs**:
   ```bash
   fly logs
   ```

---

## 🔧 Extensibility

- Swap in another segmentation model by implementing the same interface in `api/model.py`.
- Add authentication, rate-limiting or metrics (e.g. Prometheus).
- Bundle weights via S3 or HTTP and cache on startup.

---

## 🤝 Contributing

1. Fork & branch: `git checkout -b feature/awesome`
2. Code → tests → documentation
3. Open a PR, ensure CI passes.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Happy segmenting!* 🚀
