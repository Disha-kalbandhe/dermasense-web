# DermaSense

DermaSense is a monorepo for a multimodal skin disease analysis system. It combines a Next.js frontend, a FastAPI inference backend, and a PyTorch-based machine learning pipeline for training, evaluation, and experimentation.

## Project Structure

```text
.
├── backend/         # FastAPI application serving ML predictions
├── frontend/        # Next.js 16 web application (React, Tailwind)
├── ml/              # PyTorch training pipeline, notebooks, and models
└── inspect_model.py # Model inspection utility
```

## Prerequisites

Before running anything, make sure you have the following installed:

- **Python 3.10+** for the backend and ML pipeline.
- **Node.js 18+** for the frontend.
- **npm** or another compatible Node package manager.
- **Git** for cloning and version control.
- A working virtual environment workflow for Python projects.

If you plan to work with the ML pipeline, you should also have:

- Jupyter Notebook or JupyterLab.
- Enough local disk space for data, processed artifacts, and experiment outputs.

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd dermasense-web
```

If your local folder name is different, adjust the `cd` command accordingly.

---

### 2. Backend Setup

The backend is a FastAPI service that exposes inference endpoints for the model.

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend server:

```bash
uvicorn app.main:app --reload
```

The backend is where inference logic, request handling, and model-serving code live.

If your backend configuration needs environment variables, create a local `.env` file inside `backend/` and keep secrets out of version control.

---

### 3. Frontend Setup

The frontend is a Next.js application for the DermaSense user interface.

```bash
cd ../frontend
npm install
npm run dev
```

This starts the web app in development mode.

If your frontend needs API URLs or other configuration, create a local `.env.local` file inside `frontend/`.

---

### 4. ML Pipeline Setup

The `ml/` directory contains the PyTorch training code, notebooks, evaluation scripts, and supporting artifacts used for experimentation.

```bash
cd ../ml
python -m venv .venv
```

Activate the Python environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the ML dependencies:

```bash
pip install -r requirements.txt
```

To explore notebooks or run experiments interactively:

```bash
jupyter notebook
```

If you prefer JupyterLab, use:

```bash
jupyter lab
```

The ML folder is intended for training, validation, evaluation, and experimentation rather than the production frontend experience.

---

### 5. Model Inspection Utility

At the repository root, `inspect_model.py` is a helper script for inspecting or validating the model setup. Use it when you want a quick way to check model-related behavior without opening the full training flow.

## Repository Notes

This repository intentionally keeps large or machine-specific files out of version control.

You should expect the following to be ignored by `.gitignore`:

- **Python environments:** `.venv/`, `venv/`, `env/`, `ENV/`
- **Node dependencies and build output:** `node_modules/`, `.next/`
- **Secrets:** `.env`, `.env.local`, and other local environment files
- **Dataset and artifact files:** raw dataset directories, zip archives, checkpoint files, and other generated ML outputs

Common ignored ML artifacts include:

- `ml/data/raw/DATASET/`
- `ml/data/*.zip`
- `ml/checkpoints/*.pth`
- `ml/checkpoints/*.pt`

This is intentional so that the repository stays lightweight and reproducible. If you download datasets or generate trained weights locally, keep them inside the ignored paths above.

## Development Workflow

A typical workflow looks like this:

1. Work on the ML pipeline inside `ml/`.
2. Use the backend to expose inference logic.
3. Use the frontend to present predictions and analysis to users.
4. Keep large binaries, raw data, and local environment files untracked.

For code changes, prefer small, focused commits so that frontend, backend, and ML updates stay easy to review.

## Contributing

If you are adding new functionality:

- Update the relevant subdirectory first.
- Keep frontend, backend, and ML responsibilities separate.
- Avoid committing generated files, local environments, or large data artifacts.
- Document any new setup steps directly in the relevant folder README if needed.

## Notes for Contributors

- If you change backend dependencies, update `backend/requirements.txt`.
- If you change frontend dependencies or scripts, update `frontend/package.json`.
- If you add new ML experiments or training logic, keep notebooks and scripts organized inside `ml/`.
- If you add new ignored artifacts, update `.gitignore` accordingly.

## Status

This monorepo is organized for collaborative development across product UI, model serving, and machine learning research. The clean folder split is designed to keep the project maintainable as the frontend, backend, and ML pipeline evolve independently."# Taken over by Disha - $(date)" 
