# 🤖 General Health Query Chatbot

<p align="center">
  <b>🧬 AI Chatbot | 🏥 Healthcare NLP | 💊 Medical Q&A | 🧠 Mental Health</b><br><br>
  An AI-powered chatbot that answers general health-related queries using machine learning and natural language processing.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Framework-FastAPI%2FStreamlit-orange">
  <img src="https://img.shields.io/badge/NLP-Transformers-brightgreen">
  <img src="https://img.shields.io/badge/Healthcare-AI-red">
</p>

---

## 🩺 Project Overview

The **General Health Query Chatbot** is designed to understand and respond to user queries on a range of health topics such as:

* Symptoms
* Diseases
* Medications
* Mental health

Powered by machine learning, this chatbot can be accessed through a web interface using **FastAPI** or **Streamlit**.

---

## ✅ Features

* 🔍 Understands natural language health-related queries
* 💡 Provides answers using pre-trained and custom-trained NLP models
* 🧠 Handles topics across general health, medication, and mental well-being
* 🌐 Deployable via Streamlit or FastAPI for easy user access

---

## 🧾 Dataset Used

The model is trained using a combination of publicly available medical Q\&A datasets:

* 🗃️ **Medical Chatbot Dataset**
* 🧠 **Mental Health Counseling Dataset**
* 🏥 **HealthCare Chatbot Dataset**
* 🧪 **Multi-Class Medical Dataset**

These datasets include structured medical questions and their expert-verified answers.

---

## 🧠 Model Workflow

1. **Data Preprocessing**

   * Text cleaning, tokenization, stopword removal
2. **Model Training**

   * Train ML/DL models (Logistic Regression, BERT, etc.)
3. **Prediction Pipeline**

   * User enters query → NLP model processes → Generates answer
4. **Interface Deployment**

   * Streamlit or FastAPI for interaction

---

## ⚙️ Tech Stack

| Tool/Library    | Purpose                            |
| --------------- | ---------------------------------- |
| 🐍 Python       | Programming language               |
| 🤗 Transformers | Language models like BERT          |
| 🧪 Scikit-learn | Traditional ML algorithms          |
| 🧹 NLTK / spaCy | NLP preprocessing                  |
| 🌐 Streamlit    | Web interface (optional)           |
| 🚀 FastAPI      | Backend API for chatbot deployment |

---

## 🧪 Evaluation Metrics

* **Accuracy**: Correctness of predictions
* **F1-Score**: Balance between precision and recall
* **Response Relevance**: Human evaluation of answers

---

## 💻 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Aalyan-butt/General-Health-Query-Chatbot
cd health-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```



### 3. Run the Application

* **FastAPI**

```bash
uvicorn app.main:app --reload
```

* **Streamlit**

```bash
streamlit run app/interface.py
```

---

## 🧑‍⚕️ Example Usage

> **Input:** "What are the symptoms of diabetes?"
> **Response:** "Common symptoms of diabetes include excessive thirst, frequent urination, fatigue, and blurred vision."

> **Input:** "Can I take ibuprofen for a headache?"
> **Response:** "Yes, ibuprofen is commonly used for headaches, but it should be taken according to dosage guidelines."

---

## 🚀 Future Enhancements

* [ ] Integrate real-time medical API for live data
* [ ] Enable multilingual support
* [ ] Add speech-to-text for voice query interface
* [ ] Use GPT-based models for deeper conversational capabilities

---



## 🙌 Acknowledgments

* Open healthcare datasets from Kaggle and Hugging Face
* Hugging Face Transformers for powerful NLP models
* Streamlit & FastAPI for simple deployment

---

## 👨‍💻 Author

**Aalyan Riasat**
📧 [aalyanriasatali@gmail.com](mailto:your.email@example.com)
🔗  • [GitHub](https://github.com/Aalyan-butt)

---
