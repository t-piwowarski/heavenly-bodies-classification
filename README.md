# Heavenly Bodies Classification with Machine Learning
The goal is to classify celestial objects (Galaxy, Star, QSO) based on photometric and spectroscopic features from the SDSS dataset. Includes full data preprocessing, class balancing using SMOTE, and performance analysis using accuracy, confusion matrix and class-wise metrics.

---

## 📚 Dataset

The project uses the [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data), which contains **100,000 observations** from the Sloan Digital Sky Survey (SDSS).

Each sample describes a celestial object with photometric and spectroscopic features and assigns it to one of the following classes:
- **GALAXY**
- **STAR**
- **QSO (quasar)**

---

## 📂 Repository structure

heavenly-bodies-classification \ 
│ \
│── data\
│ │── star_classification.csv\
│ \
│── src\
│ │── preprocessing.py\
│ │── modeling.py\
│ │── evaluation.py\
│\
│── main.py\
│── requirements.txt\
│── README.md

---

## 🚀 Installation

1. **Clone repository:**

   ```bash
   git clone https://github.com/t-piwowarski/heavenly-bodies-classification.git
   cd text-generator
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   
- On Windows:
     
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
- On Linux/macOS:
     
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
3. **Install the required packages:**
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the main pipeline:**

   ```bash
   python main.py
   ```

   This will:

   - Load and clean the data
   - Scale and balance the features
   - Train KNN (on balanced data), Decision Tree and Random Forest (with class weights)
   - Show confusion matrices and full per-class evaluation
