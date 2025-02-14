# Stock-Price-Prediction-with-Sentiment-Analysis

## 📌 Project Overview
In this project, we present an innovative approach to stock market analysis by offering two distinctive functionalities:
1. **Real-time Sentiment Score Computation:** Utilizes VADER sentiment analysis to derive sentiment scores from news headlines, providing insights into market trends.
2. **Stock Price Prediction Using GAN & Twitter Sentiment Analysis:** Employs a Generative Adversarial Network (GAN) to analyze historical tweet data and stock prices for forecasting future stock trends.

This research-driven project aims to enrich the understanding of market dynamics, empowering investors to make data-driven decisions.

---

## 📊 Features
- **Real-time Sentiment Score Analysis** from financial news headlines.
- **Stock Price Prediction using GAN**, integrating social media sentiment analysis.
- **Data Visualization** for market trends and sentiment trends.
- **Technical Indicators Integration**, including Bollinger Bands, RSI, and MACD.
- **Flask-based Web Application** for easy accessibility.

---

## 🏗 Methodology
### 1️⃣ Sentiment Score Computation
- **Data Collection:** News headlines sourced from FinViz.
- **Preprocessing & Cleaning:** Tokenization, stopword removal, and stemming.
- **Sentiment Analysis:** VADER model to compute sentiment scores.
- **Visualization:** Hourly & daily sentiment score graphs for stock analysis.

### 2️⃣ Stock Price Prediction Using GAN
- **Data Sources:** Historical tweets (from Kaggle) & stock data (from Yahoo Finance).
- **Feature Engineering:** Incorporating sentiment scores and technical indicators.
- **Model Implementation:**
  - Generator: Gated Recurrent Unit (GRU)
  - Discriminator: Convolutional Neural Network (CNN)
- **Training & Validation:** Backpropagation with loss function optimization.
- **Output:** Predicted vs. real stock price comparison.

---

## 🚀 Installation & Execution
### Prerequisites
Ensure you have the following installed:
```bash
pip install tensorflow flask beautifulsoup4 matplotlib plotly numpy pandas urllib3 nltk gunicorn
```

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ananya-jaikumar/DM-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DM-Project
   ```
3. Modify the dataset path in `app.py` (Line 177, Line 217).
4. Run the application:
   ```bash
   python app.py
   ```
5. Open `http://localhost:5000` in your browser to access the web interface.

---

## 📌 Results
### Sentiment Score Visualization
![image](https://github.com/user-attachments/assets/3c57f728-6aaf-41bb-a80a-07ed0e8e14d3)


### Stock Price Prediction Output
![image](https://github.com/user-attachments/assets/4fb21ea5-d0df-4f57-af16-05d5a04aeea6)


---

## 📁 Dataset
- **Stock Tweets Dataset:** Contains historical tweets for sentiment analysis.
- **Stock Data (Yahoo Finance):** Provides stock price history & technical indicators.

---
## 🤝 Contributors
- **Ananya Jaikumar** - [GitHub](https://github.com/ananya-jaikumar)
- **Kunal Sharma**

---

## 📜 References
This project is inspired by research on sentiment analysis, GANs, and stock market prediction methodologies. For detailed references, check our report.

---

## 📌 Future Scope
- Enhancing the predictive model with real-time data streams.
- Incorporating more sentiment analysis models.
- Expanding stock market datasets for broader insights.

---


### ⭐ Star this repo if you found it helpful!

