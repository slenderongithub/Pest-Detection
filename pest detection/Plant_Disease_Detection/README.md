# 🔬 LeafScan — AI Plant Disease Detector

> Upload a leaf photo. Get an instant diagnosis, severity rating, and treatment recommendation — powered by deep learning.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAI](https://img.shields.io/badge/FastAI-1.0.61-00b0ff?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Model](https://img.shields.io/badge/Model-ResNet34-6366f1?style=flat)

---

## What It Does

LeafScan analyses plant leaf images and returns:

- **Disease name** — identified from 38 classes across 11 plant species
- **Confidence score** — how certain the model is
- **Severity level** — Critical / High / Medium / Low / Healthy
- **Cause** — fungal, bacterial, viral, or pest
- **Treatment tip** — actionable next steps
- **Top 3 predictions** — alternative diagnoses with probabilities
- **Scan history** — last 5 scans tracked in session

## Supported Plants

🍎 Apple · 🫐 Blueberry · 🍒 Cherry · 🌽 Corn · 🍇 Grape · 🍊 Orange · 🍑 Peach · 🫑 Pepper · 🥔 Potato · 🍓 Strawberry · 🍅 Tomato

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/slenderongithub/Pest-Detection.git
cd Pest-Detection/Plant_Disease_Detection

# 2. Install dependencies
pip install streamlit fastai==1.0.61 ipython torch torchvision pillow

# 3. Run
streamlit run app.py
```

The model file (`app/models/export_resnet34_model.pkl`) will be downloaded automatically on first run if not present.

## Tech Stack

| Component | Details |
|-----------|---------|
| Model | ResNet34 fine-tuned via FastAI |
| Dataset | [PlantVillage](https://plantvillage.psu.edu/) — 38 disease classes |
| Backend | FastAI v1 + PyTorch |
| Frontend | Streamlit |
| Deployment | Local / GCP App Engine / AWS Elastic Beanstalk |

## Project Structure

```
Plant_Disease_Detection/
├── app.py                  # Streamlit application
├── app/
│   ├── models/             # ResNet34 model weights (.pkl)
│   ├── static/             # CSS, JS, images
│   └── view/               # HTML templates (legacy Flask UI)
├── notebook/               # Experimentation notebooks (PyTorch, Keras, TF, FastAI)
├── deployment_guide/       # AWS & GCP step-by-step guides
├── requirements.txt
└── Dockerfile
```

## Deployment

See [`deployment_guide/`](deployment_guide/) for step-by-step instructions for:
- **GCP App Engine** — [`gcp_deployment.md`](deployment_guide/gcp_deployment.md)
- **AWS Elastic Beanstalk** — [`aws_deployment.md`](deployment_guide/aws_deployment.md)
- **Local Flask** — [`deployment_guide/local_flask/`](deployment_guide/local_flask/)

---

*Built at VIT-AP University as part of a research project on automated plant disease detection.*
