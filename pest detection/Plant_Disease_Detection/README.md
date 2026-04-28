# LeafScan - AI Plant Disease Detector

LeafScan is a Streamlit app for classifying plant leaf images into one of 38 PlantVillage disease classes and showing a severity level, cause, and treatment tip.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAI](https://img.shields.io/badge/FastAI-1.0.61-00b0ff?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Model](https://img.shields.io/badge/Model-ResNet34-6366f1?style=flat)

## Features

- Upload a plant leaf image and get an instant prediction
- View confidence score and top 3 predictions
- See severity, likely cause, and treatment guidance
- Works with 38 PlantVillage classes across 11 plant species
- Automatically downloads the exported model file on first run if needed

## Supported Plants

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Strawberry, and Tomato.

## Quick Start

```bash
git clone https://github.com/slenderongithub/Pest-Detection.git
cd Pest-Detection/Plant_Disease_Detection

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

If you already have the dependencies installed, you can skip the virtual environment steps and just run:

```bash
cd "/Users/slender/Developer/Codes/pest detection/Plant_Disease_Detection"
streamlit run app.py
```

## How It Works

1. Upload a leaf image.
2. The app preprocesses the image.
3. A fine-tuned ResNet34 model predicts the disease class.
4. The app maps the prediction to severity, cause, and treatment guidance.
5. The top 3 predictions are displayed for comparison.

## Model Details

- Architecture: ResNet34
- Framework: FastAI v1 + PyTorch
- Model file: `app/models/export_resnet34_model.pkl`
- Dataset: PlantVillage
- Classes: 38

The app will try to download the model automatically if the `.pkl` file is missing.

## Project Structure

```text
Plant_Disease_Detection/
├── app.py                  # Streamlit application entry point
├── app/
│   ├── models/             # Exported FastAI model files
│   ├── static/             # Images and static assets
│   └── view/               # Legacy frontend files
├── notebook/               # Training and experiment notebooks
├── deployment_guide/       # Deployment instructions
├── requirements.txt
└── Dockerfile
```

## Requirements

The project uses the dependencies in `requirements.txt`, including:

- fastai==1.0.61
- torch
- torchvision
- streamlit
- numpy
- pillow

## Troubleshooting

- If the app shows a model warning, make sure `app/models/export_resnet34_model.pkl` exists.
- If FastAI fails to import, reinstall the dependencies from `requirements.txt`.
- If you are on macOS Apple Silicon, use a compatible Python environment and reinstall the dependencies in that environment.

## Deployment

See the guides in [`deployment_guide/`](deployment_guide/) for deployment options and setup notes.

## Notes

- This repository contains both the current Streamlit app and older FastAI/Flask-era assets.
- The Streamlit app is the main entry point in `app.py`.

## License

See [LICENSE](LICENSE) for details.
