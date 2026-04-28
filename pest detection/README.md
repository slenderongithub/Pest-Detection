# Pest Detection — Automated Pest & Disease Detection with Pesticide Recommendation

A deep learning-based system that detects agricultural pests from images and recommends appropriate pesticide treatments. Built with a custom CNN trained on a multi-class pest image dataset.

## Project Structure

```
pest-detection/
├── data/
│   └── Pest_Dataset/          # Raw image dataset (9 pest categories, ~4,770 images)
├── notebooks/
│   └── Pest_Detection_Training.ipynb   # Model training and evaluation notebook
├── scripts/
│   ├── generate_figures.py    # Generates all paper figures (Fig 1–7)
│   └── generate_fig7.py       # Standalone Fig 7 generator (sample output)
├── docs/
│   ├── Pesticides_With_Agri_Guideline_Dosage.xlsx   # Pesticide reference database
│   └── paper_figures/         # Generated figures for the research paper
│       ├── Fig1_Class_Distribution.png
│       ├── Fig2_System_Pipeline.png
│       ├── Fig3_CNN_Architecture.png
│       ├── Fig4_Recommendation_Flowchart.png
│       ├── Fig5_Accuracy_Loss_Curves.png
│       ├── Fig6_Class_Proportions_Pie.png
│       └── Fig7_Sample_Output.png
├── Plant_Disease_Detection/   # Web app & deployment (Flask + FastAI ResNet34)
│   ├── app/                   # Flask server, static assets, HTML views
│   ├── notebook/              # Experimentation notebooks (PyTorch, Keras, TF, etc.)
│   ├── deployment_guide/      # AWS & GCP deployment documentation
│   ├── app.py                 # Entry point for the web application
│   └── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The `Pest_Dataset/` contains images of **9 pest categories**:

| Pest | Images |
|------|--------|
| Aphids | 2,456 |
| Ampelophaga | 458 |
| Aleurocanthus spiniferus | 414 |
| Alfalfa plant bug | 393 |
| Alfalfa weevil | 314 |
| Apolygus lucorum | 228 |
| Aphis citricola | 210 |
| Adristyrannus | 186 |
| Alfalfa seed chalcid | 111 |

## Model

- **Architecture**: Custom CNN (3× Conv2D + MaxPool blocks → Dense 128 → Softmax output)
- **Input size**: 224 × 224 × 3
- **Classes**: 9 pest categories
- **Final validation accuracy**: ~63.87% (15 epochs)

## Generating Research Figures

Run from the `scripts/` directory:

```bash
cd scripts
python generate_figures.py
```

Figures will be saved to `docs/paper_figures/`.

## Web Application

The `Plant_Disease_Detection/` directory contains the full Flask web app with a ResNet34 model (FastAI). See its own `README.md` for setup and deployment.

## Pesticide Recommendation

Detected pests are matched against `docs/Pesticides_With_Agri_Guideline_Dosage.xlsx` to provide pesticide name and recommended dosage. Recommendations are only returned when model confidence ≥ 85%.
