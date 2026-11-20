# AI Sales Forecasting System Using LSTM and PyTorch

An end-to-end AI-powered sales forecasting application that uses a Long Short-Term Memory (LSTM) neural network to predict future sales based on historical sales patterns and calendar information.

The system includes data preprocessing, exploratory data analysis, model training, evaluation, a FastAPI backend, a React dashboard, and Docker containerization.

## 1. Project Overview

The objective of this project is to forecast future retail sales using historical sales data.

The system analyzes historical sales patterns and calendar-related features to generate a 16-day sales forecast.

**Key capabilities:**
- Historical sales data analysis and visualization.
- Time-series data preprocessing.
- LSTM-based deep learning model.
- 16-day sales forecasting.
- Model evaluation against a naive baseline.
- REST API using FastAPI.
- Interactive React dashboard.
- Docker-based application deployment.

## 2. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python, JavaScript |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Recharts |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch, LSTM |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React.js, Vite, CSS |
| API Integration | Axios |
| Containerization | Docker, Docker Compose |
| Version Control | Git, GitHub |

## 3. Dataset

**Dataset:** Store Sales – Time Series Forecasting

**Source:** Kaggle

https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

The dataset contains retail sales information from Corporación Favorita, including:

- Date
- Store number
- Product family
- Sales
- Promotions

For the initial forecasting experiment, the project focuses on Store 1 and the GROCERY I product family.

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and identify sales patterns.

Activities included:

- Dataset inspection and statistical analysis.
- Missing-value and duplicate checks.
- Daily and weekly sales analysis.
- Sales trend visualization.
- Weekend versus weekday comparisons.
- Outlier analysis.
- Holiday and promotion analysis.

These observations helped guide the preprocessing and feature engineering process.

## 5. Data Preprocessing

The preprocessing pipeline includes:

- Date conversion and chronological sorting.
- Handling missing dates.
- Feature engineering.
- Sales and promotion scaling.
- Calendar feature generation.
- Chronological train, validation, and test splitting.
- Time-series sequence generation.

**Historical input features (6):**
1. Scaled sales
2. Scaled promotions
3. Day-of-week sine
4. Day-of-week cosine
5. Weekend indicator
6. Holiday indicator

**Future input features (4):**
1. Day-of-week sine
2. Day-of-week cosine
3. Weekend indicator
4. Holiday indicator

The model uses 90 historical days to predict the next 16 days.

## 6. LSTM Model Architecture

The project uses an improved LSTM architecture implemented with PyTorch.

**Model configuration:**

| Parameter | Value |
|---|---:|
| Historical input features | 6 |
| Future calendar features | 4 |
| Lookback window | 90 days |
| Forecast horizon | 16 days |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |

The model processes historical sales information through an LSTM encoder.

Future calendar features are passed through a linear projection layer. The historical context and future calendar embeddings are combined and passed through fully connected layers to generate daily sales predictions.

## 7. Model Evaluation

The LSTM model was evaluated against a naive forecasting baseline.

Recorded evaluation results from the project experiments:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive Baseline | 885.50 | 1212.20 | 42.33% |
| Improved LSTM | 353.85 | 603.59 | 13.74% |

The improved LSTM achieved lower error values than the naive baseline in the recorded experiment.

*Note: These are experiment results from the selected checkpoint. Results can vary between training runs.*

## 8. Backend – FastAPI

The backend provides REST API endpoints for generating sales forecasts.

**Available endpoints:**

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API home |
| GET | `/health` | Health check |
| POST | `/predict/` | Prediction using prepared features |
| POST | `/predict/raw` | Prediction using raw historical data |

The raw prediction endpoint accepts:
- 90 historical records.
- 16 future dates.

It returns a JSON response containing 16 predicted sales values.

**Swagger documentation:**

`http://localhost:8000/docs`

## 9. Frontend – React Dashboard

The React dashboard provides an interactive interface for using the forecasting system.

**Dashboard features:**
- JSON sales data upload.
- Forecast generation.
- Average predicted sales.
- Total predicted sales.
- Highest predicted sales.
- Historical and predicted sales chart.
- 16-day forecast table.

The frontend communicates with the FastAPI backend using Axios.

## 10. Docker Deployment

The application is containerized using Docker and Docker Compose.

The system contains two containers:

1. Backend container – FastAPI and PyTorch model.
2. Frontend container – React production build served through Nginx.

**Run the complete application:**

```bash
docker compose up --build -d
```

**Check running containers:**

```bash
docker compose ps
```

**Stop the application:**

```bash
docker compose down
```

**Application URLs:**

| Service | URL |
|---|---|
| React Dashboard | http://localhost:5173 |
| FastAPI Backend | http://localhost:8000 |
| Swagger API Docs | http://localhost:8000/docs |

## 11. Project Structure

```text
ai-sales-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── api.py
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   └── exceptions/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── Dockerfile
│   └── package.json
│
├── models/
├── artifacts/
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

## 12. Future Improvements

- Support multiple stores and product families.
- Incorporate future promotion information.
- Improve holiday-calendar consistency between training and inference.
- Add automated model retraining.
- Implement experiment tracking and model monitoring.
- Add database integration for storing forecasts.
- Deploy the application to a cloud platform.

## 13. Author

**AI Sales Forecasting System**

Developed as an end-to-end time-series forecasting and deep learning project using Python, PyTorch, FastAPI, React, and Docker.


# ==========================================================
# SUMMARY: PHASE 2
# ==========================================================
#
# WHAT THIS CODE DOES
#
# 1. load_train_data()
#    Loads the original CSV.
#
# 2. inspect_dataset()
#    Checks shape, columns, data types, missing values
#    and duplicates.
#
# 3. analyze_selected_data()
#    Filters Store 1 and GROCERY I and analyzes dates
#    and sales.
#
# 4. save_processed_data()
#    Saves the filtered CSV.
#
# 5. main()
#    Runs the complete workflow.
#
# ==========================================================
#           AI SALES FORECASTING PROJECT - PHASE STEPS
# ==========================================================
#
# Project Phase                  Libraries Used
# ----------------------------------------------------------
#
# 1. Data Loading                Pandas, pathlib
#
# 2. Data Inspection             Pandas, NumPy
#
# 3. Exploratory Data Analysis   Pandas, Matplotlib, Seaborn
#
# 4. Data Preprocessing          Pandas, NumPy, Scikit-learn
#
# 5. Baseline Model              Scikit-learn
#
# 6. LSTM Development            PyTorch
#
# 7. Model Training              PyTorch
#
# 8. Model Evaluation            Scikit-learn, Matplotlib
#
# 9. Model Persistence           PyTorch, Joblib
#
# 10. Backend Development        FastAPI, Pydantic, Uvicorn
#
# 11. Frontend Development       React, Axios
#
# 12. Testing and Deployment     Pytest, Docker, Git
#
# ==========================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================
#
# Step 1: Daily Sales Trend
# Learn how sales change over time.
#
# Step 2: Monthly Sales Analysis
# Identify which months have higher sales.
#
# Step 3: Weekly Sales Analysis
# Identify which weekdays have higher sales.
#
# Step 4: Sales Distribution
# Understand how sales values are distributed.
#
# Step 5: Outlier Analysis
# Identify unusual sales values.
#
# Step 6: Promotion Analysis
# Analyze the relationship between promotions and sales.
#
# Step 7: Seasonality Analysis
# Identify recurring sales patterns.
#
# Step 8: EDA Summary
# Summarize important findings for preprocessing.
#
# ==========================================================
<!--
summary for 01_eda.ipynb - phase 3
We can eventually provide the model with additional features such as:

| Feature             | Purpose                         |
|---------------------|---------------------------------|
| Previous sales      | Historical sales pattern        |
| Day of week         | Weekly sales pattern            |
| Month               | Monthly seasonality             |
| Day of month        | Monthly purchasing pattern      |
| Onpromotion         | Promotion impact                |
| Holiday indicator  | Holiday-related changes         |

-->


<!--

# Phase 3: What is EDA?

Imagine you own a grocery store.

You have sales records for almost five years. Every day, your store sells different quantities of groceries.

Now you want to build an AI model that predicts:

**"How much will my grocery store sell in the next 16 days?"**

Before building that AI model, you need to understand your historical sales data.

That's exactly what EDA does.

**EDA = Exploring and understanding your data before building a machine learning model.**

Think of it like a doctor examining a patient before deciding on treatment. You examine the data before deciding how to train your model.

# What exactly did you do in Phase 3?

You performed 9 main analyses.

## 1. Monthly Sales Trend

**Question:** How are my sales changing month by month?

You grouped daily sales into months and created a graph.

For example:

- January: ₹50,000
- February: ₹55,000
- March: ₹48,000

These are illustrative values, not your actual results.

**Why did you do this?**

To understand whether sales increase, decrease, or fluctuate over time.

This helps us identify patterns that an LSTM model may learn.

## 2. Sales Distribution

**Question:** What is the normal range of my daily sales?

You calculated:

| Measurement | Meaning | Your Result |
|---|---|---:|
| Mean | Average sales | 2,223 |
| Median | Middle sales value | 2,283.5 |
| Minimum | Lowest sales | 0 |
| Maximum | Highest sales | 9,065 |

**Why did you do this?**

To understand how much your store typically sells and how much sales vary.

## 3. Weekday vs Weekend Analysis

**Question:** Does my store sell more on weekdays or weekends?

Your results:

| Day Type | Average Sales |
|---|---:|
| Weekdays | 2,441 |
| Weekends | 1,678 |

**What did you discover?**

Your store has higher average sales on weekdays.

**Why is this useful?**

Because the day of the week might help predict future sales.

## 4. Average Sales by Day

**Question:** Which particular day has the highest and lowest sales?

You calculated average sales for Monday, Tuesday, Wednesday, and so on.

Your findings:

- Wednesday: Highest average sales.
- Sunday: Lowest average sales.

**Why did you do this?**

To identify weekly sales patterns.

## 5. Promotion Analysis

First, understand what promotion means.

Suppose a grocery store gives discounts on certain products.

You wanted to understand whether sales are different on promotion days.

Your results:

| Situation | Average Sales |
|---|---:|
| Without promotion | 1,873 |
| With promotion | 2,404 |

**What did you discover?**

Sales were higher on days when products were on promotion.

**Why is this useful?**

Because the number of promotional items could be an important input feature for your forecasting model.

## 6. Correlation

This is a little different from the previous analysis.

You wanted to measure the relationship between two variables:

- Sales
- Number of items on promotion

Your result:

**0.4003**

**Positive correlation**

This means that higher promotion counts tend to be associated with higher sales in your data.

Remember:

Correlation tells us whether two variables move together. It does not prove that one causes the other.

## 7. Scatter Plot

You created a graph containing many dots.

Each dot represents one day's sales and promotion count.

- X-axis: Number of items on promotion.
- Y-axis: Daily sales.

**Why did you do this?**

To visually examine the relationship between promotions and sales.

## 8. Outlier Detection

Now imagine your store normally sells around 2,000–3,000 units daily.

But on one day, sales are 9,065.

That's unusual compared with normal days.

Such unusual observations are called **outliers**.

You used the IQR method to identify them.

Your result:

**72 potential outliers.**

**Why did you do this?**

Because unusually high or low values can affect model training. We need to investigate them before deciding what to do.

## 9. Outlier Visualization

You created two graphs:

- Boxplot: Shows the distribution of sales and potential outliers.
- Sales over time: Shows when unusual sales values occurred.

**Why did you do this?**

To understand whether unusual values are random, genuine business events, or possible data problems.

We don't automatically delete them.

# The Entire Phase 3 in One Picture

## Your EDA Workflow

**Historical Sales Dataset**

Daily sales records from 2013–2017.

**Explore the Data**

- Monthly trends
- Sales distribution
- Weekday and weekend patterns

**Understand Relationships**

- Promotion vs sales
- Correlation
- Scatter plot

**Find Unusual Values**

- IQR method
- Boxplot
- Outliers over time

**EDA Findings**

Understand your data before training the AI model.

# Why is Phase 3 Important for Your LSTM Project?

Your LSTM model will learn from historical sales data to predict future sales.

But if you don't understand your data, you won't know:

- Whether sales have weekly patterns.
- Whether promotions are related to sales.
- Whether unusual values exist.
- Whether sales fluctuate over time.

EDA helps you make better decisions during the next phase.

## The Difference Between Phase 3 and Phase 4

| Phase 3 — EDA | Phase 4 — Preprocessing |
|---|---|
| Understand the data | Prepare the data |
| Find patterns | Create useful features |
| Identify outliers | Decide how to handle them |
| Study relationships | Scale numerical values |
| Visualize sales | Prepare LSTM input sequences |

-->

