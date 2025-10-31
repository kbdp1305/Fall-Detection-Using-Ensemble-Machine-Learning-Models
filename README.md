# 🧍‍♂️ TriDA: Trial-Channel Double Attention for Fall Detection Using Ensemble Machine-Learning Models  
### 🏆 **2nd Place Winner — Data Slayer 2.0 National AI & Data Science Competition (Officially Held by Telkom University, 2024)**  
### 👥 Team : *Kalah Menang Tetap Nganggur*  
**Fauzan Ihza Fajar · Krisna Bayu Dharma Putra · Akmal Muzakki Bakir**

###  Run the Colab Notebook
> Attention-enhanced fall detection that fuses **pose**, **appearance**, and **motion** features via a **TriDA** block and an **ensemble** of lightweight classifiers—built for accuracy *and* real-time inference.  

[▶️ **Open the Full Colab Notebook**](https://colab.research.google.com/drive/1_v7Fdh5uu5uUF7F-kOrLDRYVDJoq_kwj?usp=sharing)

> **Note:** The complete notebook exceeds 25 MB and is hosted on Colab.  
---

## 📘 Overview
**TriDA (Trial-Channel Double Attention)** is an **attention-enhanced fall-detection framework** that fuses **pose**, **appearance**, and **motion** information to distinguish true human falls from normal daily activities with high precision and real-time responsiveness.

This project was developed for the **Data Slayer 2.0 National Data Science and Machine Learning Competition**, **officially organized and hosted by Telkom University, Indonesia**.  
Our team, *Kalah Menang Tetap Nganggur*, earned **🏆 2nd place among 220 teams nationwide**, recognized for achieving exceptional accuracy and inference speed across unseen datasets.

---

## 🎯 Motivation
Falls remain one of the primary causes of severe injury and mortality among the elderly. Traditional vision-based systems struggle with:

- ❌ False positives caused by similar daily actions (e.g., sitting or bending)  
- ❌ Sensitivity to lighting and camera angles  
- ❌ Latency on real-time embedded devices  

**TriDA** addresses these challenges using:

- ✳️ **Dual-attention modules (CBAM)** to refine spatial and channel features  
- 🩻 **Multi-stream pose estimation (Mediapipe BlazePose + YOLOv8-Pose)**  
- 🧭 **Spine orientation vectors** for motion and directional context  
- ⚙️ **Ensemble tree models (Random Forest, XGBoost, LightGBM, CatBoost)** for robust final classification  

---

## 🧩 Methodology

### ⚙️ Workflow Pipeline
```text
Input Video  
   ↓  
Person Detection (ResNet + CBAM)  
   ↓  
Pose Estimation (Mediapipe BlazePose + YOLOv8-Pose)  
   ↓  
Spine Vector Calculation  
   ↓  
Feature Fusion via TriDA Block  
   ↓  
Ensemble Classifier → Fall / Non-Fall
```

---

### 🧠 1️⃣ Pose and Feature Extraction
| Component | Framework | Function |
|:--|:--|:--|
| **YOLOv8-Pose** | Ultralytics | Detects person and extracts 17 keypoints from frames in real time |
| **Mediapipe BlazePose GHUM Heavy** | Google Mediapipe | Produces 33-landmark skeleton with fine-grained spatial precision |
| **Spine Vector Module** | Custom | Computes spine angle and ratio to analyze postural orientation |
| **CBAM (Convolutional Block Attention Module)** | Integrated with ResNet-50 | Adds spatial and channel attention for better feature selection |

---

### 🧩 2️⃣ Feature Integration
Each frame is processed into three feature groups – pose, visual, and geometric – then merged using the **Trial-Channel Double Attention (TriDA)** block:  

\[
F_{TriDA} = CBAM(F_{CNN}) \oplus Pose(Mediapipe, YOLOv8) \oplus SpineVector
\]

The output is a unified representation fed to an ensemble of classifiers to predict fall vs non-fall events robustly under class imbalance conditions.

---

### 🧠 3️⃣ Data Augmentation
To compensate for minority class scarcity (fall samples ≈ 15 %), the team applied **ADASYN synthetic oversampling**.  
This balanced dataset improved recall and reduced bias toward non-fall activities.

| Distribution | Before | After ADASYN |
|:--|:--:|:--:|
| Fall samples | ~1 200 | ~4 000 |  
| Non-Fall samples | ~8 000 | ~8 000 |  

---

## 🧮 Modeling and Architecture

| Model | Feature Source | F1 Score | Notes |
|:--|:--|:--:|:--|
| YOLOv8-Pose Only | Pose Keypoints | 0.9232 | Baseline vision model |
| Mediapipe-Pose Only | Pose Landmarks | 0.9350 | Higher skeleton coverage |
| Mediapipe + YOLOv8 | Dual Pose Fusion | 0.9554 | Complementary features |
| **Proposed TriDA** | Multi-source Fusion + Ensemble | **0.9962** | Best model — 99.8 % acc / F1 |

**Ensemble models:** Random Forest, CatBoost, XGBoost, and LightGBM with majority voting achieved final F1 = 0.998 and accuracy = 0.998 on validation data.

---

## 🔍 Evaluation Highlights
- **Cross-validation:** 5-fold split (75 % train · 15 % val · 10 % test)  
- **Metric:** F1-score (primary), Accuracy and Precision as supporting  
- **Public Leaderboard:** 0.9969 F1 | **Private Leaderboard:** 0.9938 F1  
- **Average Inference Latency:** < 0.08 s per frame (T4 GPU)

---

## 🚀 Results and Performance
- 🧠 **TriDA outperformed all baseline models by 4–6 % F1 margin**.  
- ⚡ **Real-time inference (> 12 FPS)** achieved without hardware optimization.  
- 🩺 **Spine vector + pose fusion** significantly reduced false positives in “sitting” and “bending” actions.  
- 🧾 **Attention visualization maps** showed that CBAM focused on critical body regions (head and torso) during fall frames.

---

## 🌍 Impact
- Provides a scalable solution for **elderly care, health monitoring, and smart CCTV** applications.  
- Architecture is lightweight enough for **edge devices (Raspberry Pi + Jetson Nano)**.  
- Enables real-time alerts and analytics dashboards for nursing homes and IoT systems.

---

## ⚙️ Technical Stack
```python
Python 3.10  
OpenCV, NumPy, Pandas, Matplotlib  
PyTorch, TensorFlow, Ultralytics YOLOv8  
Mediapipe, LightGBM, XGBoost, CatBoost, Scikit-learn  
ADASYN (SMOTE variant), CBAM attention module
```

---

## 💻 Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/TriDA-FallDetection.git
cd TriDA-FallDetection
```

### 2️⃣ Run the Colab Notebook
> Attention-enhanced fall detection that fuses **pose**, **appearance**, and **motion** features via a **TriDA** block and an **ensemble** of lightweight classifiers—built for accuracy *and* real-time inference.  

[▶️ **Open the Full Colab Notebook**](https://colab.research.google.com/drive/1_v7Fdh5uu5uUF7F-kOrLDRYVDJoq_kwj?usp=sharing)

> **Note:** The complete notebook exceeds 25 MB and is hosted on Colab.  

---

## 📜 Conclusion
- ✅ **TriDA successfully integrates CBAM, pose estimation, and ensemble learning** to achieve state-of-the-art fall detection accuracy (99.8 %).  
- 🚀 **2nd place winner at Data Slayer 2.0 Competition**, officially held by **Telkom University**.  
- 💡 Demonstrates the potential of attention-based multi-stream fusion for real-time human-activity understanding.

---

## 👥 Authors
| Name | Role | Contribution |
|:--|:--|:--|
| **Krisna Bayu Dharma Putra** | Team Leader · Artificial Intelligence Engineer · Model Architect | TriDA block design, training pipeline, CBAM integration, Yolov8 pose extractor, multi-stream pose fusion|
| **Akmal Muzakki Bakir** | Artificial Intelligence Engineer | Mediapipe pose extractor and power point organizer|
| **Fauzan Ihza Fajar** | Researcher | Paper research |

📧 [linkedin.com/in/dharma-putra1305](https://linkedin.com/in/dharma-putra1305)  
🌐 [github.com/kbdp1305](https://github.com/kbdp1305)
