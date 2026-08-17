# Customer Segmentation Using K-Means

## Project Overview

This project uses **unsupervised machine learning** to discover meaningful customer segments from customer demographic and spending data.

Unlike supervised learning projects such as customer churn prediction, this project has **no target variable (`y`)**. Instead, the K-Means clustering algorithm identifies groups of customers with similar characteristics.

The project demonstrates how customer segmentation can support business decision-making, marketing, customer targeting, and personalization.

---

## Objective

The main objective is to identify distinct customer groups based on:

- Age
- Annual Income
- Spending Score

The final model identifies **6 customer segments**.

---

## Dataset

The project uses the **Mall Customers** dataset containing 200 customer records.

The dataset contains:

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Genre | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Spending score assigned to the customer |

### Features used for clustering

The clustering model uses:

```text
Age
Annual Income (k$)
Spending Score (1-100)
```

`CustomerID` was excluded because it is only an identifier.

`Genre` was not used in this version because the objective was to demonstrate clustering based on numerical behavioral/demographic features without introducing a categorical feature into the K-Means distance calculation.

---

## Machine Learning Approach

The project follows this workflow:

```text
Customer Dataset
       |
       v
Exploratory Data Analysis
       |
       v
Feature Selection
       |
       v
Feature Scaling
       |
       v
K-Means Clustering
       |
       v
Elbow Method
       |
       v
Silhouette Score
       |
       v
Final K-Means Model (K=6)
       |
       v
PCA Visualization
       |
       v
Business Interpretation
```

---

## Data Exploration

The dataset contains:

- **200 customers**
- **5 columns**
- No missing values

The numerical variables were examined using descriptive statistics and visualizations.

An initial scatter plot of Annual Income against Spending Score showed visible patterns in the customer population.

---

## Feature Scaling

Feature scaling was performed using `StandardScaler`.

This was important because the features have different numerical ranges:

- Age: approximately 18–70
- Annual Income: approximately 15–137
- Spending Score: 1–100

K-Means relies on distances between observations. Without scaling, variables with larger numerical ranges can have disproportionate influence on the clustering process.

After scaling, the transformed features had approximately:

```text
Mean = 0
Standard deviation = 1
```

---

## K-Means Clustering

K-Means was used to divide customers into groups based on similarity.

The model was initially explored with different values of K.

The final model uses:

```text
n_clusters = 6
random_state = 42
n_init = 10
```

---

## Choosing the Number of Clusters

Two methods were used to determine a suitable number of clusters.

### 1. Elbow Method

The Elbow Method evaluated K values from 2 to 10 by examining the Within-Cluster Sum of Squares (WCSS), also known as inertia.

The curve showed substantial improvements as K increased initially, followed by progressively smaller improvements around K=5–6.

### 2. Silhouette Score

Silhouette scores were calculated for K values from 2 to 10.

The highest score was obtained at:

```text
K = 6
Silhouette Score ≈ 0.4284
```

Therefore, **6 clusters** were selected for the final model.

---

## Final Model Performance

The final K-Means model achieved:

```text
Number of clusters: 6
Silhouette Score:   0.4284
Number of customers: 200
```

A silhouette score of approximately 0.43 indicates reasonable cluster structure and separation, while also showing that the customer groups are not perfectly separated.

---

## Customer Segments

The final model produced six customer segments.

| Cluster | Business Segment | Avg. Age | Avg. Income (k$) | Avg. Spending | Customers |
|---:|---|---:|---:|---:|---:|
| 0 | Mature Moderate Customers | 56.3 | 54.3 | 49.1 | 45 |
| 1 | Young Mainstream Customers | 26.8 | 57.1 | 48.1 | 39 |
| 2 | High-Income Low-Spenders | 41.9 | 88.9 | 17.0 | 33 |
| 3 | High-Value Customers | 32.7 | 86.5 | 82.1 | 39 |
| 4 | Young High-Spenders | 25.0 | 25.3 | 77.6 | 23 |
| 5 | Low-Income Low-Spenders | 45.5 | 26.3 | 19.4 | 21 |

### Business Interpretation

**Cluster 0 — Mature Moderate Customers**

Older customers with moderate income and moderate spending behavior.

**Cluster 1 — Young Mainstream Customers**

Younger customers with moderate income and moderate spending.

**Cluster 2 — High-Income Low-Spenders**

Customers with high income but relatively low spending scores. This segment may be useful for targeted engagement and personalized offers.

**Cluster 3 — High-Value Customers**

Relatively young customers with high income and very high spending scores. This is potentially the most commercially valuable segment.

**Cluster 4 — Young High-Spenders**

Younger customers with relatively low income but very high spending scores.

**Cluster 5 — Low-Income Low-Spenders**

Customers with relatively low income and low spending scores.

> Note: These business names are interpretations of the statistical clusters. K-Means itself only produces cluster IDs.

---

## PCA Visualization

Principal Component Analysis (PCA) was used to reduce the three-dimensional feature space to two dimensions for visualization.

The first two principal components explained approximately:

```text
Principal Component 1: 44.27%
Principal Component 2: 33.31%

Combined:              77.58%
```

This means the 2D PCA representation retained approximately **77.6% of the variance** in the original three-feature dataset.

PCA was used for visualization only. The K-Means model itself was trained using the scaled original features.

---

## Model Inference

The trained model can be used to assign new customers to existing segments.

Example:

```python
new_customer = pd.DataFrame(
    [[30, 90, 85]],
    columns=[
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
)

new_customer_scaled = scaler.transform(new_customer)

predicted_cluster = final_kmeans.predict(new_customer_scaled)

print(predicted_cluster)
```

The example customer was assigned to:

```text
Cluster 3
High-Value Customers
```

This demonstrates the basic inference workflow:

```text
New Customer
     |
     v
Same Scaler
     |
     v
Scaled Features
     |
     v
Trained K-Means Model
     |
     v
Cluster Assignment
     |
     v
Business Segment
```

---

## Saved Model Files

The project includes two important model artifacts:

```text
customer_segmentation_model.pkl
customer_segmentation_scaler.pkl
```

### `customer_segmentation_model.pkl`

Contains the trained K-Means model.

### `customer_segmentation_scaler.pkl`

Contains the fitted `StandardScaler`.

The scaler must be retained because new customer data must be transformed using the **same scaling process** used during model training.

---

## Project Structure

```text
Customer-Segmentation-ML/
│
├── Customer_Segmentation.ipynb
├── customer_segmentation_model.pkl
├── customer_segmentation_scaler.pkl
├── README.md
├── requirements.txt
└── data/
    └── Mall_Customers.csv
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Google Colab
- Jupyter Notebook

---

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
scikit-learn
joblib
```

---

## How to Run the Project

1. Clone the repository.
2. Install the required dependencies.
3. Open `Customer_Segmentation.ipynb`.
4. Load the dataset.
5. Run the notebook cells sequentially.
6. Review the EDA, clustering, evaluation, PCA, and segmentation results.
7. Use the saved model and scaler for inference on new customers.

---

## Important Production Consideration

The model in this repository was trained on the Mall Customers dataset.

It should **not** be assumed to be a universal customer segmentation model for other companies.

For a real company, the recommended approach is to:

1. Collect the company's customer data.
2. Select appropriate segmentation features.
3. Clean and preprocess the data.
4. Scale the features consistently.
5. Evaluate different values of K.
6. Train a company-specific clustering model.
7. Interpret the resulting clusters according to the company's business context.
8. Save the resulting model and preprocessing pipeline.
9. Use the trained model to assign new customers to the company's segments.

The methodology is reusable, but the trained clusters are specific to the data used to train the model.

---

## Future Improvements

Possible next steps include:

- Build a reusable segmentation pipeline.
- Accept customer CSV uploads.
- Automatically identify suitable numerical features.
- Compare K-Means with other clustering algorithms.
- Add hierarchical clustering and DBSCAN.
- Build a FastAPI prediction service.
- Create a web dashboard for customer segmentation.
- Automatically generate business descriptions for each segment.
- Add customer segment visualizations.
- Deploy the model as an ML API.
- Support company-specific model training.

---

## Key Learning Outcomes

This project demonstrates practical understanding of:

- Unsupervised learning
- K-Means clustering
- Feature scaling
- Distance-based learning
- Centroids
- WCSS / inertia
- Elbow Method
- Silhouette Score
- PCA
- Cluster visualization
- Business interpretation of ML results
- Model persistence using Joblib
- Applying a trained model to new data

---

## Conclusion

This project demonstrates how machine learning can discover hidden customer groups without a predefined target variable.

The final K-Means model identified **six customer segments** from 200 customers, with a final silhouette score of approximately **0.4284**.

The project also demonstrates an important transition from a machine-learning experiment to a reusable ML workflow: the trained model and scaler can be saved and later used to assign new customers to existing segments.

For production applications, however, customer segmentation models should normally be trained using the specific company's own customer data and business context.
