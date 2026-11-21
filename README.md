# Healthcare Disease Prediction – Streamlit App

This repository contains the Streamlit application for the Healthcare Disease Prediction project.

## Files
- `app.py` – Streamlit interface with two prediction pages.
- `classifier.pkl` – tuned Random Forest Classifier pipeline.
- `regressor.pkl` – tuned Random Forest Regressor pipeline.
- `feature_metadata.json` – input fields and defaults derived from the cleaned dataset.
- `requirements.txt` – Python dependencies.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload all files in this folder to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Select **Deploy**. Wait for the build to finish, then copy the public app URL.

## Model and project notes
The app uses the tuned Random Forest settings documented in the project report:
- Classification: 150 trees, max depth 20, minimum split 5, minimum leaf 2, square-root feature selection, balanced subsampling.
- Regression: 75 trees, max depth 16, minimum split 5, minimum leaf 1, max features 0.7.

The classifier artifact was fitted using the full cleaned dataset. To keep model creation practical in this environment, the regression artifact was fitted on a reproducible 20,000-row sample using the tuned settings. Therefore, the regression artifact may not reproduce the exact Stage 6 evaluation results in the report, which were obtained from the report's training/test workflow.

## Important
This is an educational demonstration only. It is not a medical device and does not provide medical diagnosis or treatment advice.

## AI assistance declaration
AI assistance was used to support application coding and deployment preparation. The project team should review the code and include this declaration in the GitHub README and report, in line with the course instructions.


### Uploading through GitHub's website (without GitHub Desktop)
The classifier is divided into three files (`classifier.part1`, `classifier.part2`, `classifier.part3`), each smaller than 25 MB. Upload all three files to the same repository folder as `app.py`. On its first launch, the app automatically joins them into `classifier.pkl` and loads the model. Do not rename or omit any part.
