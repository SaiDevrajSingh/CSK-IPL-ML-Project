# 🏏 CSK IPL Performance Predictor

A streamlined machine learning application for predicting Chennai Super Kings match outcomes in the IPL.

## ✨ Features

- **🎯 Advanced ML Model**: Optimized Random Forest classifier with 61.54% accuracy
- **🌐 Streamlit Web App**: Beautiful, interactive web interface
- **📊 Real-time Predictions**: Instant match outcome predictions
- **🏠 Home Advantage**: Considers venue and city advantages
- **📈 Historical Context**: Uses CSK's historical performance patterns

## 🚀 Quick Start

### Option 1: One-Click Start (Windows)
```bash
# Simply double-click or run:
run_streamlit.bat
```

### Option 2: Manual Start
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

### Option 3: Python Script
```bash
python run_streamlit.py
```

## 🌐 Access Your App

Once running, open your browser and go to:
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501

## 🎯 How to Use

1. **Fill in match details** in the sidebar:
   - Season year
   - Venue and city
   - Opponent team
   - Match stage and number

2. **Click "Predict Match Outcome"**

3. **View results**:
   - Win probability with confidence gauge
   - Historical context and insights
   - Head-to-head performance charts

## 🤖 Model Details

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 61.54% on test data
- **Features**: 11 engineered features including:
  - Home advantage indicators
  - Toss win/decision factors
  - Season experience and peak years
  - Opponent strength classification
  - Match importance (playoff vs league)

## 📁 Project Structure

```
ml_project/
├── streamlit_app.py              # Main Streamlit application
├── optimized_prediction.py       # Prediction pipeline
├── train_optimized_model.py      # Model training script
├── requirements_streamlit.txt     # Dependencies
├── run_streamlit.bat             # Windows launcher
├── run_streamlit.py              # Cross-platform launcher
├── .streamlit/
│   └── config.toml               # Streamlit configuration
└── artifacts/
    ├── csk_best_model_random_forest.pkl  # Trained model
    ├── venue_encoder.pkl          # Venue encoder
    ├── opponent_encoder.pkl       # Opponent encoder
    ├── model_metadata.json        # Model information
    └── feature_names.json         # Feature names
```

## 🔄 Retraining the Model

To retrain with new data:

```bash
# Update the CSV file path in train_optimized_model.py
python train_optimized_model.py
```

## ☁️ Cloud Deployment

### Streamlit Community Cloud (Recommended)
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy automatically

### Other Options
- **Heroku**: Easy deployment with git
- **Railway**: Free tier available
- **Render**: Simple cloud deployment

## 🎨 Features Showcase

- **🎯 Interactive Predictions**: Real-time match outcome forecasting
- **📊 Visual Analytics**: Probability gauges and performance charts
- **🏠 Venue Intelligence**: Home advantage calculations
- **📈 Historical Insights**: Head-to-head records and trends
- **🎨 Professional UI**: CSK-themed design with responsive layout

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Data**: Pandas, NumPy
- **Visualization**: Plotly, Altair
- **Deployment**: Python, Docker-ready

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 61.54% |
| Precision | 66.67% |
| Recall | 57.14% |
| F1-Score | 61.54% |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

---

**🏏 Ready to predict CSK's next victory? Start the app and begin forecasting!**
