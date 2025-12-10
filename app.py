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
st.title("🔮 Tes Kepribadian MBTI Berbasis AI")
st.write("Jawab 4 pertanyaan berikut. Sistem akan menganalisis jawabanmu menggunakan AI.")

q1 = st.text_area(
    "1️⃣ Saat berada di keramaian atau banyak interaksi sosial, bagaimana perasaanmu setelahnya?"
)
q2 = st.text_area(
    "2️⃣ Saat mempelajari sesuatu yang baru, kamu lebih fokus pada detail nyata atau ide dan kemungkinan?"
)
q3 = st.text_area(
    "3️⃣ Ketika mengambil keputusan penting, apa yang paling memengaruhimu?"
)
q4 = st.text_area(
    "4️⃣ Dalam menjalani hari-hari, kamu lebih nyaman dengan rencana jelas atau fleksibel?"
)

if st.button("✨ Lihat Hasil"):
    if not all(len(q) > 10 for q in [q1, q2, q3, q4]):
        st.warning("⚠️ Tolong isi semua jawaban dengan cukup jelas (minimal 10 karakter).")
    else:
        e_i, ei_score = predict_dimension(q1, ei_vec, ei_model, "I", "E")
        s_n, sn_score = predict_dimension(q2, sn_vec, sn_model, "N", "S")
        t_f, tf_score = predict_dimension(q3, tf_vec, tf_model, "F", "T")
        j_p, jp_score = predict_dimension(q4, jp_vec, jp_model, "P", "J")

        mbti = f"{e_i}{s_n}{t_f}{j_p}"

        st.subheader(f"🎯 Tipe Kepribadian Kamu: **{mbti}**")

        st.write("### 📊 Kecenderungan:")
        st.progress(ei_score)
        st.write(f"**{e_i}** lebih dominan")

        st.progress(sn_score)
        st.write(f"**{s_n}** lebih dominan")

        st.progress(tf_score)
        st.write(f"**{t_f}** lebih dominan")

        st.progress(jp_score)
        st.write(f"**{j_p}** lebih dominan")

        st.success("✅ Analisis selesai. Ini adalah estimasi kepribadian berdasarkan jawabanmu.")

