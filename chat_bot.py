import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

# === New imports for AI Chatbot ===
from langchain_google_genai import GoogleGenerativeAI
import warnings

# === Your existing Streamlit Interface ===
st.set_page_config(page_title="HealthCare ChatBot", layout="centered")
st.title("🩺 HealthCare ChatBot - Disease Prediction and Q&A")

# Load data
@st.cache_data
def load_data():
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')
    return training, testing

training, testing = load_data()
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Cross-validation score
scores = cross_val_score(clf, x_test, y_test, cv=3)

@st.cache_data
def load_master_data():
    severity, description, precautions, faq = {}, {}, {}, {}

    with open('MasterData/symptom_severity.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].isdigit():
                severity[row[0].strip()] = int(row[1])

    with open('MasterData/symptom_Description.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                description[row[0].strip()] = row[1].strip()

    with open('MasterData/symptom_precaution.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 5:
                precautions[row[0].strip()] = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]

    with open('MasterData/faq_data.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                faq[row[0].lower()] = row[1]

    return severity, description, precautions, faq

severity_dict, description_dict, precaution_dict, faq_dict = load_master_data()

# Helper functions remain the same (check_pattern, sec_predict, predict_disease)...

def check_pattern(dis_list, inp):
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return (1, pred_list) if pred_list else (0, [])

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])

def predict_disease(symptom, days):
    tree_ = clf.tree_
    feature_name = [cols[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    reduced_data = training.groupby(training['prognosis']).max()
    symptoms_present = []

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            val = 1 if name == symptom else 0
            if val <= threshold:
                return recurse(tree_.children_left[node])
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node])
        else:
            disease = le.inverse_transform([tree_.value[node].argmax()])[0]
            symptoms_given = cols[np.atleast_1d(reduced_data.loc[disease].values[0]).nonzero()[0]]
            symptoms_exp = []
            for s in symptoms_given:
                if st.radio(f"Do you also have {s}?", ["no", "yes"], key=s) == "yes":
                    symptoms_exp.append(s)

            second_pred = sec_predict(symptoms_exp)
            diagnosis = f"{disease}"
            if disease != second_pred[0]:
                diagnosis += f" or {second_pred[0]}"
            calc = (sum(severity_dict.get(s, 0) for s in symptoms_exp) * days) / (len(symptoms_exp)+1)
            advice = "⚠️ Consult a doctor." if calc > 13 else "✅ Might not be serious, but take precautions."

            return diagnosis, symptoms_exp, advice

    return recurse(0)

# Sidebar - user info and model score + mode selection
with st.sidebar:
    st.header("👤 User Information")
    name = st.text_input("Your name", placeholder="John Doe")
    st.markdown("### 🧠 Model Accuracy")
    st.success(f"Decision Tree Accuracy (CV): {round(scores.mean() * 100, 2)}%")

    # NEW: Select mode - Symptom Checker or Medical Chat
    mode = st.radio("Choose Mode", ["Symptom Checker", "Medical Chat"])

# === Medical Chat Setup ===
# Suppress warnings for AI model
warnings.filterwarnings('ignore')

API_KEY = "AIzaSyAKLRLAnQ4ng50t87Wcn8bU-TMxs7r71wQ"  # <-- Replace with your actual Google API Key

# Initialize GoogleGenerativeAI
try:
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        generation_config={"temperature":0.3, "max_output_tokens":300},
        google_api_key=API_KEY
    )
except Exception as e:
    st.error(f"Error initializing AI model: {e}")
    llm = None

def generate_ai_response(user_question):
    if not llm:
        return "AI model is not available."
    system_prompt = (
        "You are an AI medical assistant. Provide detailed, factual, and "
        "informative responses to medical inquiries based on existing medical knowledge. "
        "Avoid direct diagnosis and recommend consulting a healthcare professional."
    )
    full_prompt = f"System: {system_prompt}\nUser: {user_question}\nAI:"
    try:
        response = llm.invoke(full_prompt)
        return response
    except Exception as e:
        return f"Error: {e}"

# --- Main Interface: Switch between modes ---
if mode == "Symptom Checker":
    st.subheader("💬 Symptom Checker")
    symptom_input = st.text_input("Enter a symptom (e.g. headache, cough):")

    if symptom_input:
        conf, pred_symptoms = check_pattern(cols, symptom_input)
        if conf:
            chosen_symptom = st.selectbox("Choose the closest match:", pred_symptoms)
            num_days = st.slider("How many days have you had this symptom?", 1, 30, 3)

            if st.button("🧬 Predict Disease"):
                with st.spinner("Analyzing..."):
                    diagnosis, matched_symptoms, advice = predict_disease(chosen_symptom, num_days)
                    st.subheader(f"👋 Hello, {name or 'User'}!")
                    st.success(f"🦠 You may have: **{diagnosis}**")
                    st.info(advice)

                    if diagnosis in description_dict:
                        st.markdown("#### 📝 Description")
                        st.write(description_dict[diagnosis])

                    if diagnosis in precaution_dict:
                        st.markdown("#### 🛡️ Precautions")
                        for i, step in enumerate(precaution_dict[diagnosis]):
                            st.markdown(f"{i+1}. {step}")
        else:
            st.warning("❗ No matching symptoms found. Please try again.")

elif mode == "Medical Chat":
    st.subheader("🧠 Ask a health-related question")

    user_query = st.text_input("Type your question (e.g., What is diabetes? How to treat anxiety?):")

    if user_query:
        lower_query = user_query.lower().strip()
        found = False
        for key, response in faq_dict.items():
            if key in lower_query:
                st.markdown("#### 💡 Answer")
                st.write(response)
                found = True
                break
        if not found:
            with st.spinner("Searching for an answer..."):
                ai_answer = generate_ai_response(user_query)
                st.markdown("#### 🤖 AI Medical Assistant says:")
                st.write(ai_answer)
