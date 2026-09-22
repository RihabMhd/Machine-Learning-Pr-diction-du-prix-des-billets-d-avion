#  Flight Price Prediction

Machine Learning project for predicting the price of an airline ticket from flight and booking characteristics.

## Project Overview

This project aims to develop a supervised Machine Learning model capable of predicting the price of an airline ticket based on different characteristics related to the flight and the booking.

The project follows the complete Machine Learning workflow:

**Exploration → Analysis → Preprocessing → Modeling → Evaluation → Export → Deployment**

The final objective is to produce a reproducible and exploitable prediction pipeline that can later be integrated into a Streamlit application.

---

##  Objectives

The main objectives of this project are to:

* Explore and understand a real-world dataset.
* Identify the nature and role of the variables.
* Analyze the relationships between flight characteristics and ticket prices.
* Prepare the data for Machine Learning.
* Build a preprocessing pipeline using Scikit-learn.
* Compare several regression models.
* Evaluate model performance and generalization.
* Estimate the uncertainty associated with the model error.
* Export the final Machine Learning pipeline.
* Develop a Streamlit application for price prediction.
* Optionally containerize the application using Docker.

---

##  Dataset

The dataset contains information about airline flights and their prices.

### Main variables

| Variable           | Description                      | Type / Role           |
| ------------------ | -------------------------------- | --------------------- |
| `airline`          | Airline company                  | Categorical           |
| `flight`           | Flight identifier                | Categorical           |
| `source_city`      | Departure city                   | Categorical           |
| `departure_time`   | Departure time category          | Categorical           |
| `stops`            | Number of stops                  | Ordinal / categorical |
| `arrival_time`     | Arrival time category            | Categorical           |
| `destination_city` | Destination city                 | Categorical           |
| `class`            | Travel class                     | Ordinal               |
| `duration`         | Flight duration                  | Quantitative          |
| `days_left`        | Number of days before the flight | Quantitative          |
| `price`            | Ticket price                     | **Target variable**   |
| `Unnamed: 0`       | Dataset identifier               | Identifier            |

The target variable of the project is:

```text
price
```

---

##  Project Workflow

### 1. Data Exploration

The first stage consists of:

* Loading the dataset.
* Checking the number of rows and columns.
* Inspecting the first observations.
* Checking data types.
* Checking missing values.
* Computing descriptive statistics.

### 2. Variable Analysis

Variables are classified according to their nature:

* Categorical variables
* Ordinal variables
* Quantitative variables
* Target variable

### 3. Univariate Analysis

Categorical variables are analyzed using:

* Frequencies
* Percentages
* Category distributions
* Bar charts

Quantitative variables are analyzed using:

* Descriptive statistics
* Histograms
* Boxplots
* Skewness

### 4. Multivariate Analysis

The project studies relationships between explanatory variables and ticket price, including:

* Flight duration → price
* Airline → price
* Number of stops → price
* Days before flight → price

The observed relationships will be interpreted based on the analysis results.

### 5. Preprocessing

The preprocessing stage will include:

* Missing-value handling
* Treatment of ordinal variables
* Encoding categorical variables
* Feature selection
* Separation of `X` and `y`
* Train/test split

The transformations will be integrated into a Scikit-learn `Pipeline` and fitted only on the training data to avoid data leakage.

### 6. Modeling

Several regression models will be compared:

* `DummyRegressor`
* `LinearRegression`
* `Ridge`
* `RandomForestRegressor`

For Ridge, several values of `alpha` will be tested.

### 7. Evaluation

The models will be evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

A 5-fold cross-validation will also be performed on the training set.

### 8. Final Model

After comparing the models, a final model will be selected and evaluated on the test set using:

* MAE
* RMSE
* R²

### 9. Uncertainty Analysis

A 95% confidence interval around the MAE will be estimated from the absolute errors obtained on the test set.

The calculation will include:

* MAE
* Standard deviation of the errors
* Standard error
* 95% confidence interval

### 10. Model Export

The complete preprocessing and prediction pipeline will be exported as a `.joblib` file.

A metadata file will also describe the model and the variables expected by the pipeline.

### 11. Streamlit Application

A Streamlit application will allow users to enter flight characteristics and obtain an estimated ticket price.

The application will use the exported Machine Learning pipeline directly.

### 12. Docker — Bonus

As an optional step, the Streamlit application can be containerized using Docker.

```text
Machine Learning Pipeline
          ↓
      Streamlit
          ↓
        Docker
```

---

##  Project Structure

```text
flight-price-ml/
│
├── data/
│   └── Clean_Dataset.csv
│
├── notebooks/
│   └── flight_price_prediction.ipynb
│
├── artifacts/
│   ├── flight_price_model.joblib
│   └── flight_price_model.meta.json
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

This structure follows the organization proposed in the project specification.

---

##  Technologies

The project uses:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Git / GitHub

Docker may be used as an optional bonus.

---

##  Installation

Clone the repository:

```bash
git clone <repository-url>
cd flight-price-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

##  Usage

### Run the notebook

Open:

```text
notebooks/flight_price_prediction.ipynb
```

and execute the cells sequentially.

### Run the Streamlit application

Once the application is available:

```bash
streamlit run app/app.py
```

---

## 📅 Project Schedule

The project is planned over five working days, from **21/09/2026 to 25/09/2026**.

| Day   | Main tasks                                         |
| ----- | -------------------------------------------------- |
| Day 1 | Exploration, classification, univariate analysis   |
| Day 2 | Multivariate analysis, preprocessing               |
| Day 3 | Regression models, cross-validation                |
| Day 4 | Final evaluation, confidence interval, export      |
| Day 5 | Streamlit, Git, README, presentation, Docker bonus |

---

## 👤 Author

**Rihab Mahdi**

Machine Learning project — YouCode UM6P

---

## 📚 Deliverables

The final project will contain:

1. A structured Machine Learning notebook.
2. An exported `.joblib` pipeline.
3. A metadata file.
4. A functional Streamlit application.
5. A Git repository containing the project and documentation.

These correspond to the expected project deliverables.
