
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

st.set_page_config(page_title="Sentimen Kualitas Udara", layout="wide")


POSITIF = {
    "baik", "bersih", "sehat", "segar", "aman", "lega", "membaik", "turun",
    "pulih", "sejuk", "bantuan", "peduli", "bagus", "cerah", "jernih",
    "hijau", "solusi", "berhasil", "terkendali", "nyaman", "selamat",
}
NEGATIF = {
    "buruk", "polusi", "kotor", "bahaya", "berbahaya", "parah", "sesak",
    "batuk", "ispa", "asap", "kabut", "kabur", "tercemar", "pencemaran",
    "tidak sehat", "naik", "memburuk", "rusak", "sakit", "mengganggu",
    "pekat", "debu", "menyengat", "khawatir", "gawat",
}
WARNA = {"positif": "Greens", "negatif": "Reds", "netral": "Blues"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}


# ---------------------------------------------------------------- Fungsi
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URL
    text = re.sub(r"@\w+", " ", text)                  # mention
    text = re.sub(r"#\w+", " ", text)                  # hashtag
    text = re.sub(r"\d+", " ", text)                   # angka
    text = re.sub(r"[^a-zA-Z\s]", " ", text)           # simbol/emoji
    text = text.lower()                                # case folding
    return re.sub(r"\s+", " ", text).strip()           # spasi berlebih


def label_sentiment(text: str) -> str:
    tokens = text.split()
    joined = " " + text + " "
    pos = sum(t in POSITIF for t in tokens)
    neg = sum(t in NEGATIF for t in tokens)
    # frasa dua kata
    for w in POSITIF | NEGATIF:
        if " " in w and f" {w} " in joined:
            if w in POSITIF:
                pos += 1
            else:
                neg += 1
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


@st.cache_data
def load_and_label(path_or_file):
    df = pd.read_csv(path_or_file)
    df = df[["created_at", "username", "full_text"]].dropna(subset=["full_text"])
    df["clean_text"] = df["full_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df["sentimen"] = df["clean_text"].apply(label_sentiment)
    return df


@st.cache_resource
def train_model(texts: tuple, labels: tuple):
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts), list(labels), test_size=0.2, random_state=42,
        stratify=labels if min(Counter(labels).values()) >= 2 else None,
    )
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                        solver="adam", max_iter=300, random_state=42)
    mlp.fit(Xtr, y_train)
    y_pred = mlp.predict(Xte)
    return vec, mlp, y_test, y_pred


def make_wordcloud(text, cmap):
    if not text.strip():
        return None
    return WordCloud(width=700, height=400, background_color="white",
                     colormap=cmap, max_words=100).generate(text)


# ---------------------------------------------------------------- UI
st.title("📊 Analisis Sentimen Kualitas Udara")
st.caption("Pipeline: Cleansing → Pelabelan rule-based → TF-IDF → Neural Network (MLP)")

uploaded = st.sidebar.file_uploader("Upload tweet_data.csv", type="csv")
source = uploaded if uploaded else "tweet_data.csv"
try:
    df = load_and_label(source)
except FileNotFoundError:
    st.error("File tweet_data.csv tidak ditemukan. Jalankan crawl_tweet.py atau upload CSV di sidebar.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dataset", "Distribusi Sentimen", "WordCloud", "Evaluasi Model", "Prediksi Teks Baru"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah tweet", len(df))
    c2.metric("Jumlah kolom", df.shape[1])
    c3.metric("Username unik", df["username"].nunique())
    st.dataframe(df[["created_at", "username", "full_text", "clean_text", "sentimen"]],
                 use_container_width=True)
    st.download_button("Unduh dataset berlabel (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "tweet_data_berlabel.csv", "text/csv")

with tab2:
    counts = df["sentimen"].value_counts().reindex(["positif", "netral", "negatif"]).fillna(0)
    colors = ["#2e9e4f", "#4a7fd1", "#d64545"]
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_title("Jumlah Tweet per Sentimen")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors)
        ax.set_title("Proporsi Sentimen")
        st.pyplot(fig)
    st.table(counts.astype(int).rename("jumlah").to_frame())

with tab3:
    cols = st.columns(3)
    for col, s in zip(cols, ["positif", "netral", "negatif"]):
        with col:
            st.subheader(f"{EMOJI[s]} {s.capitalize()}")
            wc = make_wordcloud(" ".join(df.loc[df["sentimen"] == s, "clean_text"]), WARNA[s])
            if wc is None:
                st.info("Tidak ada data.")
            else:
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

vec, mlp, y_test, y_pred = train_model(tuple(df["clean_text"]), tuple(df["sentimen"]))

with tab4:
    labels_order = [c for c in ["positif", "netral", "negatif"] if c in set(df["sentimen"])]
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{acc:.3f}")
    m2.metric("Presisi", f"{p:.3f}")
    m3.metric("Recall", f"{r:.3f}")
    m4.metric("F1-Score", f"{f:.3f}")
    st.caption("Presisi, recall, dan F1 memakai rata-rata weighted. "
               f"Data uji: {len(y_test)} tweet (split 80:20).")

    pc, rc, fc, sc = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_order, zero_division=0)
    tabel = pd.DataFrame({"presisi": pc, "recall": rc, "f1-score": fc, "support": sc},
                         index=labels_order)
    st.dataframe(tabel.style.format({"presisi": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}))

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots()
        tabel[["presisi", "recall", "f1-score"]].plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Metrik per Kelas")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    with g2:
        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels_order, yticklabels=labels_order, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

with tab5:
    teks = st.text_area("Masukkan teks:", "Kualitas udara hari ini membaik, langit cerah dan terasa segar")
    if st.button("Prediksi"):
        bersih = clean_text(teks)
        if not bersih:
            st.warning("Teks kosong setelah dibersihkan.")
        else:
            proba = mlp.predict_proba(vec.transform([bersih]))[0]
            hasil = mlp.classes_[int(np.argmax(proba))]
            st.markdown(f"### {EMOJI[hasil]} Sentimen: **{hasil.upper()}**")
            st.write("Teks bersih:", bersih)
            st.dataframe(pd.DataFrame({"kelas": mlp.classes_, "probabilitas": proba})
                         .sort_values("probabilitas", ascending=False))
            st.bar_chart(pd.Series(proba, index=mlp.classes_))
