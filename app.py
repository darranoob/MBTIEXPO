import streamlit as st
import joblib

st.set_page_config(
    page_title="MBTI Personality Test (AI)",
    page_icon="🧠",
    layout="centered"
)

# Load models
ei_vec, ei_model = joblib.load("ei_model.pkl")
sn_vec, sn_model = joblib.load("sn_model.pkl")
tf_vec, tf_model = joblib.load("tf_model.pkl")
jp_vec, jp_model = joblib.load("jp_model.pkl")

# Helper function
def predict_dimension(text, vectorizer, model, pos, neg):
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0]
    return (pos, prob[1]) if prob[1] >= 0.5 else (neg, prob[0])

# UI
st.title("🔮 MBTI Personality Test with AI")
st.write("Answer this 4 question. System will analyst your answer with AI.")

q1 = st.text_area(
    "1️⃣ When you're in the crowd or having much social interaction, what are you feeling after that?"
)
q2 = st.text_area(
    "2️⃣ When you learn something new, where do you foucs? (example: Idea/Fact/Detail/Etc)"
)
q3 = st.text_area(
    "3️⃣ When you're making a decision, what affects you? (example: Other people feelings/Logic/Etc)"
)
q4 = st.text_area(
    "4️⃣ Everyday which one more comfort you, a structured plan or a flexible flow?"
)

if st.button("✨ Lihat Hasil"):
    if not all(len(q) > 10 for q in [q1, q2, q3, q4]):
        st.warning("⚠️ Please input the answer more than 10 characters")
    else:
        e_i, ei_score = predict_dimension(q1, ei_vec, ei_model, "I", "E")
        s_n, sn_score = predict_dimension(q2, sn_vec, sn_model, "N", "S")
        t_f, tf_score = predict_dimension(q3, tf_vec, tf_model, "F", "T")
        j_p, jp_score = predict_dimension(q4, jp_vec, jp_model, "P", "J")

        mbti = f"{e_i}{s_n}{t_f}{j_p}"

        st.subheader(f"🎯 Your Type: **{mbti}**")

        st.write("### 📊 Tendency:")
        st.progress(ei_score)
        st.write(f"**{e_i}** more dominant")

        st.progress(sn_score)
        st.write(f"**{s_n}** more dominant")

        st.progress(tf_score)
        st.write(f"**{t_f}** more dominant")

        st.progress(jp_score)
        st.write(f"**{j_p}** more dominant")

        st.success("✅ Analysis done. This is your result based on your answer.")


