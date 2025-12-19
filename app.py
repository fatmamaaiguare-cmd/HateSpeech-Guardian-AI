import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.nn.functional import softmax
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="HateSpeech Guardian AI",
    page_icon="🛡️",
    layout="wide"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .hate-tag {
        background-color: #ffcccc;
        color: #cc0000;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        border: 1px solid #ff9999;
    }
    .safe-text {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    model_path = r"C:\Users\hp\OneDrive\Desktop\HateSpeech Guardian AI\final_hate_model_optimized"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le dossier modèle est introuvable ici : {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"🔴 ERREUR : Impossible de charger le modèle.\n\n{e}")
    st.stop()

# --- FONCTION DE PRÉDICTION ---
def predict_text(text, threshold):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = softmax(logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    temp_results = []
    for i, token in enumerate(tokens):
        if token in ["<s>", "</s>"]: continue
        score = probs[i][1].item()
        clean_token = token.replace(" ", "")
        if token.startswith(" "): clean_token = " " + clean_token
        temp_results.append({"word": clean_token, "score": score})

    final_results = []
    n = len(temp_results)
    
    for i in range(n):
        word_obj = temp_results[i]
        score = word_obj["score"]
        is_hate = False
        
        if score > threshold:
            is_hate = True
        else:
            left_toxic = (i > 0) and (temp_results[i-1]["score"] > threshold)
            right_toxic = (i < n-1) and (temp_results[i+1]["score"] > threshold)
            if (left_toxic or right_toxic) and score > 0.30:
                is_hate = True
                
        clean_word_strip = word_obj["word"].strip().lower()
        if clean_word_strip in ["this", "that", "is", "are", "a", "the", ".", ",", "to", "in"]:
            if score < 0.85: is_hate = False

        final_results.append((word_obj["word"], is_hate, score))
        
    return final_results

# --- INTERFACE ---
with st.sidebar:
    st.title("⚙️ Paramètres")
    threshold = st.slider("Sensibilité", 0.0, 1.0, 0.65, 0.05)
    st.info(f"Seuil : **{int(threshold*100)}%**")

st.title("🛡️ HateSpeech Guardian AI")

tab1, tab2, tab3 = st.tabs(["🔍 Analyse Temps Réel", "📂 Audit CSV", "📊 Analytics"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area("Texte à analyser :", height=100)
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Analyser", use_container_width=True): # Correction warning
            pass # Le bouton déclenche le rerun
    
    if text_input:
        res = predict_text(text_input, threshold)
        html = "<div style='line-height:2.5; font-size:18px; border:1px solid #eee; padding:20px; border-radius:10px;'>"
        for w, h, s in res:
            if h: html += f"<span class='hate-tag' title='{int(s*100)}%'>{w}</span>"
            else: html += f"<span class='safe-text'>{w}</span>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

with tab2:
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Colonne Texte :", df.columns)
        if st.button("Lancer l'Audit"):
            prog = st.progress(0)
            audit_data = []
            for i, row in df.iterrows():
                txt = str(row[target_col])
                pred = predict_text(txt, threshold)
                is_toxic = any([p[1] for p in pred])
                words = [p[0].strip() for p in pred if p[1]]
                audit_data.append({"Texte": txt, "Statut": "🔴 HAINE" if is_toxic else "🟢 OK", "Mots": ", ".join(set(words))})
                prog.progress((i+1)/len(df))
            
            res_df = pd.DataFrame(audit_data)
            st.session_state['audit_results'] = res_df
            # CORRECTION DU WARNING : use_container_width -> width="stretch"
            st.dataframe(res_df, width=1000) 
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger", csv, "rapport.csv", "text/csv")

with tab3:
    if 'audit_results' in st.session_state:
        data = st.session_state['audit_results']
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(data, names='Statut', title="Toxicité")
            st.plotly_chart(fig, use_container_width=True) # Plotly gère encore use_container_width
    else:
        st.info("Veuillez lancer un audit d'abord.")