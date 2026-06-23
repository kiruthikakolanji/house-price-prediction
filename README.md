# house-price-prediction
AI-powered house price prediction web app using Random Forest
# 🏠 House Price Prediction

An ML-powered web app that predicts California house prices based on location, income, and housing features — built using Python, Scikit-learn, and Streamlit.

## 🚀 Live Demo
[Add your deployed app link here once deployed]

---

## 📌 Project Overview

A Random Forest regression model trained on the **California Housing Dataset** (20,640 houses) that predicts house prices based on 8 features including median income, house age, rooms, and location coordinates. Deployed as an interactive Streamlit web app with sliders for real-time price prediction.

---

## 🧠 Development Process — Step by Step

### Step 1: Setup & Get the Dataset
- Installed required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`
- Loaded the **California Housing Dataset** directly from scikit-learn using `fetch_california_housing()` — no manual download needed
- Converted to a pandas DataFrame with 20,640 rows and 9 columns

### Step 2: Exploratory Data Analysis (EDA)
- Used `df.describe()` for a statistical summary of all features
- Confirmed **0 missing values** across all columns — perfectly clean dataset
- Created **Chart 1 (Price Distribution)** using seaborn `histplot` — most houses priced between $100,000–$200,000
- Created **Chart 2 (Correlation Heatmap)** using seaborn `heatmap` — discovered `MedInc` (median income) has the strongest correlation with price (0.69)
- Saved both charts as PNG files for documentation

### Step 3: Build & Train the Model
- Separated features (X = 8 columns) and target (y = Price)
- Split data 80/20 into training (16,512 samples) and testing (4,128 samples) using `train_test_split`
- Chose **Random Forest Regressor** over Linear Regression for better accuracy on non-linear housing data
- Trained model with 100 decision trees (`n_estimators=100`)
- Evaluated using MAE, RMSE, and R² score

### Step 4: Save the Model
- Saved trained model using `pickle`
- Original file was **144MB** — exceeded GitHub's 100MB per-file limit
- Compressed using Python's `gzip` library → reduced to **30MB** (79% smaller)
- Final saved file: `house_price_model.pkl.gz`

### Step 5: Build the Streamlit Web App
- Built `app.py` with 8 interactive sliders (one per feature)
- Used `st.slider` for continuous numeric inputs and `st.number_input` for population
- Loaded the compressed model using `gzip.open()` + `pickle.load()`
- Converted user inputs into a numpy array → fed to model → displayed prediction as a formatted dollar amount

### Step 6: Push to GitHub
- Used Git command line to push all files including the 30MB compressed model
- Resolved merge conflicts between local files and browser-uploaded files using `git pull --allow-unrelated-histories -X ours`

### Step 7: Deploy
- Deployed on **Streamlit Cloud**, connected directly to the GitHub repository
- Verified live app works with real-time house price predictions

---

## 📈 Model Performance

| Metric | Score | Meaning |
|--------|-------|---------|
| **R² Score** | **0.8046** | Model is 80% accurate |
| **MAE** | **0.3277** | Average error of ~$32,770 |
| **RMSE** | **0.5060** | Slightly higher due to penalizing large errors |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🐼 Pandas | Data loading and manipulation |
| 🔢 NumPy | Numerical operations and array handling |
| 🤖 Scikit-learn | Random Forest model, train/test split, evaluation metrics |
| 📊 Matplotlib & Seaborn | EDA charts (price distribution, correlation heatmap) |
| 🗜️ Gzip + Pickle | Model compression and saving |
| 🎈 Streamlit | Interactive web app interface |
| 🔧 Git & GitHub | Version control and deployment |

---

## 📊 Dataset

[California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) — built into scikit-learn, 20,640 California houses with 8 features:

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in the area |
| `HouseAge` | Age of the house |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Population in the area |
| `AveOccup` | Average occupants per household |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

---

## 💻 How to Run Locally

1. Clone this repository
```bash
git clone https://github.com/kiruthikakolanji/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

| File | Description |
|---|---|
| 📄 `app.py` | Streamlit web app — sliders + prediction logic |
| 📄 `train.py` | Model training script (EDA → model → save) |
| 📦 `house_price_model.pkl.gz` | Compressed trained Random Forest model |
| 📊 `correlation_heatmap.png` | Feature correlation heatmap chart |
| 📊 `price_distribution.png` | House price distribution chart |
| 📋 `requirements.txt` | Python dependencies |
| 📖 `README.md` | Project documentation |

---

## 🎯 Future Improvements
- Add neighborhood name input instead of raw coordinates
- Try XGBoost or Gradient Boosting for higher accuracy
- Add feature importance chart showing which factors affect price most

---

## 👤 Author
Kiruthika Kolanji – [LinkedIn Profile Link]
