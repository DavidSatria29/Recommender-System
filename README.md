# Course Recommender System Capstone

## 🚀 Project Overview

This project addresses the challenge of course discovery on online learning platforms by developing and evaluating a comprehensive course recommender system. It explores two primary methodologies: **content-based filtering**, which recommends courses based on textual similarity, and **collaborative filtering**, which leverages user rating data. Within collaborative filtering, several algorithms were implemented and compared, including K-Means clustering, NMF, and a deep learning (ANN) model. The results demonstrated that while content-based methods are effective for topical suggestions, the deep learning model was significantly more accurate in predicting user ratings, leading to the conclusion that a hybrid system would provide the most robust solution.

---

## 🛠️ Models Implemented

This project explores a variety of recommendation techniques to find the most effective approach:

* **Content-Based Filtering**: Recommends items based on the similarity of their content (e.g., course descriptions) using TF-IDF and cosine similarity.
* **Collaborative Filtering - Clustering-Based**: Groups users with similar interests using K-Means clustering on their profile vectors (which were reduced in dimension using PCA). Recommendations are based on popular courses within a user's cluster.
* **Collaborative Filtering - Memory-Based**:
    * **User-Based & Item-Based KNN**: Predicts ratings by finding the K-Nearest Neighbors (either users or items) based on rating patterns, using the `surprise` library.
* **Collaborative Filtering - Model-Based (Matrix Factorization)**:
    * **NMF**: Decomposes the user-item rating matrix into latent user and item feature matrices to predict ratings.
    * **ANN with Embeddings**: A deep learning model that learns user and item embeddings and uses dense layers to predict ratings.
    * **Regression on Embeddings**: Uses the embeddings from the pre-trained ANN model as features for traditional regression models (Ridge and Lasso) to predict ratings.

---

## 📊 Model Performance & Key Findings

The collaborative filtering models were evaluated based on their **Root Mean Square Error (RMSE)** for rating prediction, where a lower value is better. The Artificial Neural Network (ANN) model demonstrated significantly superior performance.

| Model | RMSE |
| :--- | :--- |
| **ANN with Embeddings** | **0.4303** |
| Ridge/Lasso Regression on Embeddings &nbsp; &nbsp; &nbsp; | 0.8138 |
| NMF (Matrix Factorization) | 1.2860 |
| KNN (User & Item-Based) | 1.2890 |

* **Key Insight**: The deep learning (ANN) approach, which learns complex non-linear relationships, is the most accurate for predicting user ratings in this dataset. Using embeddings from the ANN as features for simpler regression models also yields a significant performance boost over traditional CF methods.

---

## 💻 Technologies Used

* **Language**: Python
* **Libraries**:
    * Pandas & NumPy for data manipulation
    * Scikit-learn for K-Means, PCA, Ridge/Lasso Regression, and evaluation metrics
    * Keras (TensorFlow) for the deep learning model
    * Surprise for KNN and NMF implementations
    * Matplotlib & Seaborn for data visualization
* **Environment**: Jupyter Notebook

---

## 📂 Repository Content

* `lab_jupyter_content_course_similarity.ipynb`: Notebook for the Content-Based filtering model.
* `lab_jupyter_content_clustering.ipynb`: Notebook for the Clustering-Based collaborative filtering model.
* `lab_jupyter_cf_knn.ipynb`: Notebook for the KNN-Based collaborative filtering model.
* `lab_jupyter_cf_nmf.ipynb`: Notebook for the NMF-Based collaborative filtering model.
* `lab_jupyter_cf_ann.ipynb`: Notebook for the baseline Deep Learning (ANN) model.
* `lab_jupyter_cf_regression_w_embeddings.ipynb`: Notebook for the regression models using pre-trained embeddings.
* `ml-capstone-template-coursera.pdf`: The final capstone project report summarizing all methods and findings.

---

## 🔮 Future Work

Based on the project's findings, the following next steps are recommended:

1.  **Develop a Hybrid System**: Combine the strengths of the content-based model with the high-accuracy collaborative filtering ANN model.
2.  **Deployment**: Package the best-performing model into a REST API or a simple web application for real-world use.
3.  **Explore Advanced Models**: Investigate more complex deep learning architectures (e.g., Transformers) to potentially capture more nuanced user preferences.
