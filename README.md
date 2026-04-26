---
title: CardioAI – ECG Explorer
sdk: streamlit
app_file: app.py
python_version: "3.10"
---

# CardioAI Streamlit ECG Explorer

CardioAI Streamlit ECG Explorer is a lightweight research and educational prototype for viewing ECG inputs, running model inference, and presenting prediction and explanation outputs through an interactive web interface.

This repository contains the **user-facing Streamlit application** for the CardioAI project. It is kept separate from the main CardioAI research repository so that the deployment-oriented interface remains distinct from the core experimental workflow and thesis artefacts.

## Overview

The application was developed to support:

- ECG input viewing and basic interaction
- prediction display for supported CardioAI tasks
- explanation visualization for model auditing and demonstration
- research and educational presentation of the CardioAI workflow

Within the current project scope, the app is intended as a **prototype interface** and **not** as a clinical decision-support system.

## Repository Structure

```text
CardioAI_streamlit/
├── .devcontainer/
├── assets/
├── demo/
├── .gitignore
├── README.md
├── app.py
├── cardioai_infer.py
└── requirements.txt
````

### Main contents

* `app.py`
  Main Streamlit entry point for the application.

* `cardioai_infer.py`
  Inference and ECG-processing utilities used by the app.

* `assets/`
  Static assets used by the interface.

* `demo/`
  Demo files and supporting example materials.

* `requirements.txt`
  Python dependencies required to run the app.

## Relationship to the Main CardioAI Repository

This repository contains the **Streamlit demonstration layer** only.

The separate **main CardioAI repository** contains the research workflow and thesis artefacts, including:

* preprocessing and tokenization notebooks
* model experimentation and evaluation workflow
* explainability-related analysis
* figures, tables, and reproducibility documentation

This separation helps keep:

* the **research workflow repository** focused on experimentation and artefact reporting, and
* the **Streamlit repository** focused on interactive use and app deployment.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit app with:

```bash
streamlit run app.py
```

## Intended Use

This app is provided for:

* research demonstration
* educational use
* interactive presentation of CardioAI outputs

It is **not intended for clinical deployment or medical decision-making**.

## Notes

* The app may rely on external model files, assets, or demo resources depending on the deployment setup.
* Raw ECG datasets are not redistributed through this repository unless explicitly included as demo-safe sample files.
* Configuration and model-loading behavior should remain aligned with the CardioAI project scope described in the accompanying research repository.

## Disclaimer

CardioAI Streamlit ECG Explorer is a research and educational prototype. The predictions, explanations, and visual outputs produced by this application are not clinically validated and must not be used as a substitute for professional medical interpretation.
