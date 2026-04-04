# 🏥 Privacy-Preserving Healthcare AI using SplitFed

## 🚀 Overview

This project implements a hybrid distributed learning system combining Federated Learning (FL) and Split Learning (SL) to enable secure, privacy-preserving AI across multiple hospitals.

The system ensures that raw patient data never leaves local hospitals while still enabling collaborative model training.

---

## 🧠 Problem Statement

Healthcare data is highly sensitive and cannot be centralized due to:

* Privacy regulations (HIPAA, GDPR)
* Security risks
* Data ownership constraints

Traditional ML fails in such environments.

---

## 💡 Proposed Solution

We propose a hybrid **SplitFed architecture** that:

* Uses Federated Learning for distributed training
* Uses Split Learning to reduce client computation and enhance privacy
* Achieves high accuracy while preserving data confidentiality

---

## 🏗️ System Architecture

![Architecture](./assets/architecture.png)

### Flow:

1. Clients (hospitals) train locally
2. Split Learning shares intermediate activations
3. Server completes forward/backward pass
4. Federated Averaging aggregates model weights

---

## ⚙️ Tech Stack

### Backend:

* Python
* TensorFlow / Keras
* Flower (Federated Learning)
* NumPy, Pandas, Scikit-learn

### Frontend:

* React (Vite)
* Tailwind CSS
* Recharts
* Framer Motion

---

## 📊 Results

| Model       | Accuracy   |
| ----------- | ---------- |
| Centralized | 80.25%     |
| Federated   | 66.43%     |
| SplitFed    | **82.07%** |

---

## 📈 Key Insights

* Federated Learning suffers from non-IID data
* Split Learning improves generalization
* SplitFed combines both to recover accuracy

---

## ▶️ How to Run

### Backend

```bash
python src/pipelines/train_eval_pipeline.py
python src/pipelines/federated_pipeline.py
python src/pipelines/splitfed_pipeline.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 📂 Project Structure

```
src/
data/
clients/
plots/
frontend/
models/
```

---

## 🔐 Privacy Features

* No raw data sharing
* Distributed learning
* Secure aggregation (conceptual)
* Split computation across client/server

---

## 📉 Limitations

* Communication overhead
* Non-IID data challenges
* Synthetic data noise

---

## 🔮 Future Scope

* Differential Privacy (DP)
* Secure Multi-Party Computation (SMPC)
* Blockchain-based auditing

---

## 🤖 Development Note

This project was developed using structured AI-assisted engineering with approximately 20–25 iterative prompts for debugging, optimization, and system design.

---

## 👨‍💻 Author

Vedanth Dama
B.Tech CSE (AI & ML)

---

## ⭐ Star this repo if you found it useful!
