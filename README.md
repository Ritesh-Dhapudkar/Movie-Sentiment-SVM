# IMDb Movie Review Sentiment Analysis using SVM

## 📌 Project Overview
This project classifies movie reviews as **Positive** or **Negative** using Machine Learning. I used a **Support Vector Machine (SVM)** model and performed text preprocessing to handle challenges like negation (e.g., "not good").

## 🛠️ How I Built It
1.  **Data Collection:** Used a dataset of movie reviews.
2.  **Text Preprocessing:** - Removed HTML tags and special characters.
    - Used **Porter Stemming** to reduce words to their roots.
    - **Negation Handling:** Customized the stopword list to keep words like "not" and "no" so the model understands negative context.
3.  **Vectorization:** Converted text to numbers using **TF-IDF** (Unigrams and Bigrams).
4.  **Model Training:** Compared Linear, RBF, and Polynomial kernels. The **Linear Kernel** performed best.

## 🚀 Key Features
- **Interactive Predictor:** A Jupyter widget interface to test any review.
- **Improved Accuracy:** By keeping "not" in the text, the model correctly identifies "not good" as Negative.

## 📂 File Structure
- `1_Training.ipynb`: The notebook where I cleaned data and trained the SVM.
- `2_Predictor.ipynb`: The "API" notebook with the input box for testing.
- `svm_model.joblib`: The trained model weights.
- `requirements.txt`: The libraries needed to run this project.

## 💻 How to Run
1. Clone this repository.
2. Install libraries: `pip install -r requirements.txt`
3. Open the notebooks in Jupyter and run all cells.