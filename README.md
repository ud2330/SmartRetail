# **Smart Retail Analytics System**

A Flask-based retail analytics platform that processes survey data from **100+ customers** to predict stock-outs, forecast spending, segment shoppers, recommend offers, and analyze retail trends. The system supports CSV uploads, automated preprocessing, interactive dashboards, and generates actionable insights for retail decision-making.

---

## **Features**

### **User Authentication**

* Signup, Login, Sessions
* User-specific dashboard

### **Dataset Upload & Processing**

* Upload customer survey dataset (CSV)
* Auto-handling of missing values, label encoding, numeric processing
* Dynamic detection of feature types (categorical, numerical, datetime)

### **Predictive Retail Analytics**

* **Stock-Out Prediction**
* **Monthly Spend Forecasting**
* **Offer Recommendation (Multi-Label)**
* **Trend Analysis & Behavior Insights**

### **Customer Segmentation**

* K-Means clustering
* Automatically identifies **4 unique customer segments**
* Business-friendly descriptions + characteristics

### **Category Optimization**

* Insights on frequently purchased categories
* Offer and pricing strategy recommendations
* Consumer behavior interpretation

---

## **Tech Stack**

* **Python, Flask**
* **Scikit-learn (Random Forest, K-Means, Linear Regression)**
* **Pandas, NumPy**
* **MongoDB + PyMongo**
* **HTML, CSS, JavaScript**

---

## **Clone the Repository**

```bash
git clone https://github.com/ud2330/SmartRetail.git
cd SmartRetail
```

---

## **Setup**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Configure MongoDB**

Use either:

* Local MongoDB
* MongoDB Atlas (cloud)

Default Database: `smartretail`


### **3. Start the Application**

```bash
python app.py
```

### **4. Visit the App**

```
http://127.0.0.1:5000
```

---

## **Security Notes**

* Use a strong `SECRET_KEY`
* Never store passwords as plain text (use hashing in production)
* Use authenticated MongoDB users
* Enable HTTPS when deployed
* Validate all user inputs


