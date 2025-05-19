# 🔐 Network Intrusion Detection System (NIDS) Using Machine Learning

## 📘 Overview

This project implements a **Network Intrusion Detection System (NIDS)** using supervised machine learning techniques. The system is designed to detect and classify malicious traffic within a computer network, helping protect against threats such as DoS, probe, R2L, and U2R attacks.

By training on publicly available datasets such as **NSL-KDD**, **UNSW-NB15**, or **CIC-IDS2017**, this system can distinguish between **normal and anomalous behavior** in network traffic with high accuracy.

---

## 🎯 Objectives

* Detect intrusions in real-time or offline network data.
* Classify attacks into categories (e.g., DoS, Probe, etc.).
* Compare and evaluate the performance of multiple ML models.
* Provide insights into the most significant network features for detection.

---

## 📂 Dataset

Supported and tested on the following datasets:

* [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
* [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
* [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

**Common Features:**

* Duration, protocol, source/destination IP and port
* Bytes transferred
* Flags, services, connection state
* Attack label (normal or type of attack)

---

## 🛠️ Technologies Used

* **Python 3.x**
* **Pandas**, **NumPy** – Data processing
* **Scikit-learn** – Machine learning models
* **Matplotlib**, **Seaborn** – Visualization
* **Jupyter Notebook** – Development and experimentation

---

## 🤖 Machine Learning Models

Implemented and compared:

* **Logistic Regression**
* **Random Forest**
* **Decision Tree**
* **K-Nearest Neighbors**
* **Support Vector Machine**
* **Gradient Boosting**
* (Optional) Deep Learning with Keras/TensorFlow

---

## ⚙️ Workflow

1. **Data Preprocessing**

   * Handle missing values
   * Encode categorical features
   * Normalize or standardize numerical values
   * Feature selection using correlation, importance, or PCA

2. **Model Training & Evaluation**

   * Split into training/testing sets (time-aware if needed)
   * Train using supervised models
   * Evaluate using:

     * Accuracy
     * Precision
     * Recall
     * F1-Score
     * ROC-AUC

3. **Visualization**

   * Confusion matrix
   * ROC curves
   * Feature importance graphs

---

## 📈 Results (Sample Table)

| Model             | Accuracy | Precision | Recall | F1-Score |
| ----------------- | -------- | --------- | ------ | -------- |
| Random Forest     | 0.97     | 0.96      | 0.97   | 0.96     |
| Gradient Boosting | 0.98     | 0.97      | 0.98   | 0.97     |
| SVM               | 0.93     | 0.92      | 0.91   | 0.91     |

---

## 🛡️ Detection Categories (NSL-KDD Example)

* **DoS**: Denial of Service attacks
* **R2L**: Remote to Local
* **U2R**: User to Root
* **Probe**: Surveillance and probing
* **Normal**: Legitimate traffic

---

## 📌 Future Improvements

* Integrate deep learning (LSTM, CNN) for sequence-based detection.
* Create real-time detection using packet sniffing (e.g., Scapy).
* Build a web dashboard for live monitoring and alerting.
* Deploy as an API (Flask/FastAPI) or microservice.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo, submit pull requests, or open issues for improvements or questions.

---

## 🙏 Acknowledgements

* NSL-KDD & UNSW-NB15 dataset creators
* Scikit-learn team and open-source contributors
* Researchers in intrusion detection and cybersecurity

---
