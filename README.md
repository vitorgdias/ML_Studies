# Machine Learning Model Explorer 
Part of the Study from the YouTube - Tell Me Why (https://www.youtube.com/watch?v=oz_rZ92Tmls&list=PLvlkVRRKOYFR6_LmNcJliicNan2TYeFO2)

An interactive web application built with Streamlit to explore, compare, and understand the behavior of various machine learning models on real-world regression and classification tasks.

## 🎯 Project Goal

The primary objective of this project is to serve as a hands-on educational tool. It provides a clear and intuitive interface to demystify the process of training and evaluating different ML algorithms, making it easier to grasp key concepts by applying them to simple, relatable datasets.

---

## ✨ Features

This application is divided into two main modules:

### 1. 🍺 Regression: Beer Prediction

This module tackles a regression problem where the goal is to predict the beer type.

* **Dataset:** `dados_cerveja.xlsx`
* **Features (Input):**
    * Beer temperature (°C)
    * Beer glass type
    * Beer foam
    * Beer color
* **Target (Output):**
    * Beer class
* **Models Implemented:**
    * Linear Regression
    * Decision Tree Regressor
    * Random Forest Regressor
    * K-Neighbors Regressor (KNN)
    * Support Vector Regressor (SVR)

### 2. 🍎 Classification: Fruit Classifier

This module addresses a classic classification problem: identifying a fruit based on its physical characteristics.

* **Dataset:** `dados_frutas.xlsx`
* **Features (Input):**
    * Color (e.g., Red, Yellow, Green)
    * Shape (e.g., Round, Oblong)
    * Texture (e.g., Smooth, Rough)
    * Taste (Sweet)
* **Target (Output):**
    * Fruit Name (e.g., Apple, Banana, Orange)
* **Models Implemented:**
    * Linear Regression
    * Decision Tree Regressor
    * Random Forest Regressor
    * K-Neighbors Regressor (KNN)
    * Support Vector Regressor (SVR)
      
---

## 🛠️ Tech Stack

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web application.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn:** For implementing and training machine learning models.
* **Plotly:** For generating interactive visualizations like the confusion matrix.

---

Work In Progress...

## 🚀 How to Run the Project

To run this application on your local machine, please follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/vitorgdias/ml_studies.git](https://github.com/vitorgdias/ml_studies.git)
cd ml_studies
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
(Note: You will need to create a requirements.txt file. See below.)
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```
A new tab should open in your web browser with the application running.
