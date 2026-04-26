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
