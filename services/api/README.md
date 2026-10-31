# Stetho API Service

FastAPI backend service for Stetho acoustic condition monitoring.

## Local Development

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -e services/api
```

Run the server:

```powershell
uvicorn stetho_api.main:app --reload --port 8000
```

