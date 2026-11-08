# Wall-Following Robot Navigation — Move Classifier

ML Assignment 2 · M.Tech (AIML) · BITS Pilani WILP

**Live app:** <STREAMLIT_APP_URL>
**GitHub repo:** https://github.com/AdityaJogia/robot-nav-classifier

---

## a. Problem statement

A mobile robot that follows the walls of a room has to decide its next move from what its
ultrasound sensors see. The goal of this project is to **predict the robot's movement command
(one of 4 classes) from a single snapshot of its 24 ultrasound readings**. Five classical
classifiers are trained, compared on six metrics, and served through an interactive
Streamlit app. The app scores the held-out test data using the saved models and does not retrain them.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository — [Wall-Following Robot Navigation Data (#194)](https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data) |
| Robot | SCITOS G5, navigating clockwise around a room, sampled at 9 readings/sec |
| File used | `sensor_readings_24.data` (the full 24-sensor version) |
| Instances | 5,456 (no duplicates, no missing values) |
| Features | 24 numeric: `US1`…`US24`, distance in metres from ultrasound sensors placed every 15° around the robot (`US1` = front, 180°; `US13` = back, 0°). Values range about 0.34–5.03 m (about 5 m means nothing in range). |
| Target | `Action` — 4 classes |
| Licence | CC BY 4.0 |

Class distribution (imbalanced):

| Action | Count | Share |
|---|---|---|
| Move-Forward | 2,205 | 40.4 % |
| Sharp-Right-Turn | 2,097 | 38.4 % |
| Slight-Right-Turn | 826 | 15.1 % |
| Slight-Left-Turn | 328 | 6.0 % |

**Preprocessing:** stratified 80/20 train/test split (`random_state=26`), so the rare
`Slight-Left-Turn` class appears in both splits. A `StandardScaler` is fitted on the training data only and
bundled with each classifier in a scikit-learn `Pipeline`. The 20% test split is saved as
`test_data.csv` (1,092 rows) and is what gets uploaded to the app.

**Metric notes:** this is a multi-class problem, so Precision, Recall and F1 are **macro-averaged**
(each class weighted equally, so the small turn classes count). AUC is **one-vs-rest macro AUC**.

## c. GitHub repository

https://github.com/AdityaJogia/robot-nav-classifier

```
robot-nav-classifier/
├── app.py                 # Streamlit app (loads saved models, never retrains)
├── train_models.py        # trains + evaluates the 5 models, writes model/ and test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv          # held-out 20% test split (24 features + Action)
├── data/
│   ├── sensor_readings_24.data   # raw UCI file
│   └── Wall-following.names      # UCI documentation
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.json
```

## d. Models used

All five models are trained on the same training split and evaluated on the same test split.

| Model | Configuration |
|---|---|
| Logistic Regression | multinomial, `C=1.0`, `max_iter=3000` |
| Decision Tree | `max_depth=12`, `min_samples_leaf=2` |
| K-Nearest Neighbour | `k=5`, distance-weighted |
| Naive Bayes | Gaussian NB |
| Support Vector Machine | RBF kernel, `C=10`, `gamma="scale"`, probability outputs on |

### Comparison table (test set, 1,092 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7097 | 0.9129 | 0.6898 | 0.6358 | 0.6566 | 0.5560 |
| Decision Tree | 0.9908 | 0.9931 | 0.9901 | 0.9850 | 0.9875 | 0.9862 |
| KNN | 0.8974 | 0.9706 | 0.8950 | 0.8831 | 0.8887 | 0.8450 |
| Naive Bayes | 0.5302 | 0.8268 | 0.5451 | 0.6629 | 0.5524 | 0.3792 |
| Support Vector Machine | 0.9359 | 0.9875 | 0.9494 | 0.9217 | 0.9347 | 0.9028 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Moderate (71% accuracy, MCC 0.56). Its decision boundaries are linear in the sensor distances, but the robot's moves depend on threshold-style conditions such as "the wall on the right is closer than X". A linear model cannot express that well. It does worst on the two minority classes (recall about 0.50 for Slight-Left and 0.52 for Slight-Right). Its AUC of 0.91 is much better than its accuracy, so the class ranking is reasonable even though the hard decisions are often wrong. |
| Decision Tree | Best model (99.1% accuracy, MCC 0.986). The labels come from a rule-based wall-following controller, and axis-aligned splits on a few sensors reproduce those rules almost exactly. A depth-10 tree is enough. Most of the importance falls on a handful of right-side and front-right sensors (US15, US19, US12, US20, US18), the ones that measure distance to the wall being followed. Recall stays at or above 0.97 even for the rare Slight-Left-Turn class. |
| KNN | Good (89.7%, MCC 0.85). Scaled distances capture the local similarity between sensor snapshots well. However, all 24 sensors count equally in the distance, including rear sensors that barely affect the decision. That noise blurs the neighbourhoods and leaves it about 9 points behind the tree. |
| Naive Bayes | Weakest (53% accuracy, MCC 0.38). The readings are far from Gaussian: they are capped at 5.0 m, skewed and often bimodal, and neighbouring sensors are correlated, which breaks the independence assumption. It heavily over-predicts the turn classes (399 Slight-Right predictions against about 165 true ones), which is why its macro Recall (0.66) is higher than its Precision (0.55) and Accuracy. |
| Support Vector Machine | Second best (93.6%, MCC 0.90, AUC 0.99). The RBF kernel models the non-linear boundaries that Logistic Regression misses, and recall is above 0.88 for every class. It still smooths over the sharp thresholds that a tree captures exactly. |
| **Overall winner** | **Decision Tree.** It is highest on all six metrics, it is fast, and it is interpretable. That suits this dataset, whose labels were generated by a rule-based controller. SVM is a strong second if smoother decision boundaries are preferred. |

## Streamlit app features

- **CSV upload** for the test data, with column validation and a sample-file download button
- **Model selection** dropdown for all five saved models
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) computed live on the uploaded file
- **Confusion matrix** heatmap and a per-class **classification report**
- A **Compare all models** tab with a side-by-side table, the best model by MCC, and a chart

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

## Reference

Freire, A. L., Barreto, G. A., Veloso, M., & Varela, A. T. (2009). Wall-Following Robot Navigation Data. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/194
