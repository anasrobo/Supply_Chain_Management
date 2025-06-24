# Supply Chain Management — Demand Forecasting with ML

> Demand forecasting for a makeup supply chain using a neural network.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 🚀 Project Overview

This repository contains a full demand forecasting solution for a fashion & beauty startup’s makeup supply chain. By predicting **“Number of products sold”**, the model helps optimize inventory, reduce stockouts, and enhance supply-chain efficiency.

### Key Steps

1. **Data Preparation & EDA**
2. **Neural-Network Model** (128 → 64 → 32 layers)
3. **Training & Validation**
4. **Evaluation** (MSE + scatter plot)
5. **Model Export & Inference**

## 📁 Repository Structure

```
.
├── data/
│   └── supply_chain_data.csv        # Raw dataset
├── notebooks/
│   └── demand_forecasting.ipynb     # Colab-ready notebook
├── models/
│   └── demand_forecasting_model.keras  # Trained Keras model
└── README.md                        # Project overview
```

## 🛠️ Setup & Usage

1. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow
   ```

2. **Place your data:**

   Drop `supply_chain_data.csv` into the `data/` folder. Ensure it includes:

   * Product Type, SKU, Price, Availability
   * Number of products sold (target)
   * Revenue generated, Stock levels, Lead times, etc.

3. **Open the notebook:**

   Launch `notebooks/demand_forecasting.ipynb` in Google Colab or Jupyter:

   * Run the upload cell to load your CSV
   * Execute data cleaning, encoding, and feature engineering
   * Train the neural network model
   * Evaluate test MSE and view the True vs. Predicted scatter plot
   * Save the trained model in native `.keras` format

4. **Review outputs:**

   * Look for the printed Test MSE in the notebook output
   * Inspect visualization cells for model performance insights

## 🔄 Next Steps

* 🔧 Perform hyperparameter tuning (e.g., with KerasTuner)
* 📈 Experiment with advanced models (LSTM, XGBoost)
* 🚀 Add a Streamlit dashboard for interactive insights (in progress)

---

> Built with ❤️ by Anas

