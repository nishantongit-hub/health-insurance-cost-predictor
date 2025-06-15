# 🏥 Health Insurance Cost Predictor

A machine‑learning solution that estimates individual health insurance charges from demographic and lifestyle features.

## 📊 Key Features
- **Predictive model** trained on the public *Medical Cost Personal Dataset*.
- **Input factors:** age, BMI, gender, smoking status, region (and optional sum insured).
- **Visual outputs:** scatter plot of actual vs. predicted charges and residual distribution.

## 🛠 Tech Stack
| Tool / Library | Purpose |
| -------------- | ------- |
| Python 3.8+    | Core language |
| pandas, NumPy  | Data handling |
| scikit‑learn   | ML pipeline |
| matplotlib, seaborn | Visualization |
| joblib         | Model persistence |

## 📁 Project Structure
```
health-insurance-cost-predictor/
├─ health_insurance_cost_predictor.py   # Main script
├─ insurance.csv                        # Dataset (user supplied)
├─ insurance_model.pkl                  # Saved model (generated)
├─ predicted_vs_actual.png              # Output graphic
├─ residual_plot.png                    # Output graphic
├─ requirements.txt                     # Dependencies
└─ README.md                            # Documentation
```

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/health-insurance-cost-predictor.git
cd health-insurance-cost-predictor

# Install dependencies
pip install -r requirements.txt

# Add dataset (insurance.csv) then run:
python health_insurance_cost_predictor.py
```

## 📈 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🔮 Future Enhancements
- Integrate advanced models (Random Forest, XGBoost, Neural Nets)
- Deploy as a Streamlit web app or REST API
- Incorporate additional health and financial features
- Add SHAP for interpretability

## 📚 References
- Kaggle *Medical Cost Personal* Dataset  
- Pedregosa, F. *et al.* “Scikit‑learn: Machine Learning in Python.” *JMLR* (2011)  
- Lundberg, S.M. & Lee, S.‑I. “A Unified Approach to Interpreting Model Predictions.” *NeurIPS* (2017)

---

*Created with ❤️ to make health insurance pricing transparent.*
