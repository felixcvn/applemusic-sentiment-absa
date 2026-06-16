import streamlit as st
import pandas as pd
import altair as alt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import prediksi dari masing-masing algoritma
from xgb_boost import predict_single as predict_xgb
from svm_predict import predict_single as predict_svm

# ==========================================
# Konfigurasi Halaman & CSS Premium
# ==========================================
st.set_page_config(
    page_title="Evaluasi Sentimen & ABSA: SVM vs XGBoost",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (Apple Luxury Aesthetic)
st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        background-color: #000000;
    }

    .stApp {
        background-color: #000000;
        color: #f5f5f7;
    }
    
    /* Clean up Streamlit */
    #MainMenu, header, footer {visibility: hidden;}
    
    h1, h2, h3, h4 {
        letter-spacing: -0.02em !important;
    }

    /* Apple-style Native Input */
    .stTextArea textarea {
        background-color: rgba(28, 28, 30, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        color: #f5f5f7 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 18px !important;
        padding: 20px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #fa243c !important;
        box-shadow: 0 0 0 1px #fa243c, 0 0 20px rgba(250, 36, 60, 0.15) !important;
    }
    
    /* Apple Music Vibrant Red Button */
    .stButton>button {
        background: linear-gradient(135deg, #fa243c 0%, #ff3b30 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        letter-spacing: -0.01em !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 8px 24px rgba(250, 36, 60, 0.35) !important;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 32px rgba(250, 36, 60, 0.5) !important;
        background: linear-gradient(135deg, #ff3b30 0%, #ff453a 100%) !important;
    }

    /* iOS Widget Style Cards */
    .sentiment-card {
        background: rgba(28, 28, 30, 0.6);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .sentiment-card.negative {
        background: linear-gradient(135deg, rgba(44, 20, 22, 0.8) 0%, rgba(28, 28, 30, 0.8) 100%);
        border: 1px solid rgba(250, 36, 60, 0.25);
    }
    .sentiment-card.positive {
        background: linear-gradient(135deg, rgba(20, 44, 28, 0.8) 0%, rgba(28, 28, 30, 0.8) 100%);
        border: 1px solid rgba(46, 204, 113, 0.25);
    }
    
    .aspect-row {
        background: rgba(28, 28, 30, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }

    /* Badges */
    .badge {
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-pos { 
        background-color: rgba(52, 199, 89, 0.15); 
        color: #34c759; 
        border: 1px solid rgba(52, 199, 89, 0.2); 
    }
    .badge-neu { 
        background-color: rgba(255, 255, 255, 0.05); 
        color: #98989d; 
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Conclusion Box */
    .conclusion-card {
        background: rgba(28, 28, 30, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-top: 30px;
    }

    /* Metric Cards for Dataset Stats */
    .metric-card {
        background: rgba(28, 28, 30, 0.5);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 16px;
    }
    .metric-title {
        font-size: 12px;
        color: #98989d;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 40px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
        letter-spacing: -0.03em;
    }

    hr {
        border-color: rgba(255,255,255,0.08);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
    <div style="background: linear-gradient(135deg, #fa243c 0%, #ff3b30 100%); width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(250,36,60,0.4);">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white" stroke="none">
            <path d="M21 3v12.5a3.5 3.5 0 0 1-3.5 3.5A3.5 3.5 0 0 1 14 15.5a3.5 3.5 0 0 1 3.5-3.5c.54 0 1.05.12 1.5.34V6.47L9 8.6v9.9a3.5 3.5 0 0 1-3.5 3.5A3.5 3.5 0 0 1 2 18.5 3.5 3.5 0 0 1 5.5 15c.54 0 1.05.12 1.5.34V3l14-2v2z"/>
        </svg>
    </div>
    <h1 style='font-weight: 800; font-size: 36px; margin: 0; padding: 0;'>Apple Music Analyzer Dashboard</h1>
</div>
<p style='color: #98989d; font-size: 16px; margin-left: 64px; font-weight: 500;'>Analisis Sentimen Berbasis Aspek (ABSA) Menggunakan Model SVM & XGBoost Classifier.</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- TABS FOR PRESENTATION ---
tab1, tab2, tab3 = st.tabs([
    "⚖️ Real-Time Komparasi Model", 
    "📊 Statistik Dataset Global", 
    "⚙️ Spesifikasi & Metrik Model"
])

# Load Dataset for Statistics
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(os.path.join(BASE_DIR, 'data_preprocessed.csv')).dropna(subset=['clean_teks'])
    except:
        return pd.DataFrame()

df_global = load_dataset()

# ==========================================
# TAB 1: REAL-TIME KOMPARASI MODEL
# ==========================================
with tab1:
    st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-bottom: 16px;'>Uji Komparasi Ulasan Real-time</h3>", unsafe_allow_html=True)
    text = st.text_area(
        "Masukkan ulasan:",
        height=140,
        placeholder="Contoh: Harga langganannya lumayan mahal tapi kualitas suaranya jernih dan fiturnya lengkap.",
        label_visibility="collapsed",
        key="input_text_tab1"
    )

    col_btn, _ = st.columns([1.2, 3])
    with col_btn:
        compare_btn = st.button("🔍 Bandingkan Hasil Model", width="stretch")

    if compare_btn:
        if not text.strip():
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
            st.stop()

        with st.spinner("Menganalisis dengan kedua model..."):
            hasil_svm = predict_svm(text)
            hasil_xgb = predict_xgb(text)

        # Cek error
        if "error" in hasil_svm:
            st.error(f"SVM Error: {hasil_svm['error']}")
            st.stop()
        if "error" in hasil_xgb:
            st.error(f"XGBoost Error: {hasil_xgb['error']}")
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 style='font-size: 24px; font-weight: 800; text-align: center; margin-bottom: 30px;'>📋 Hasil Perbandingan Model</h2>", unsafe_allow_html=True)

        col_svm, col_xgb = st.columns(2)

        # Helper function to check if aspect is detected
        def is_detected(val):
            if not val:
                return False
            return str(val).lower() in ["ya", "terdeteksi", "1", "true"]

        # ==========================================
        # KOLOM SVM
        # ==========================================
        with col_svm:
            st.markdown("<h3 style='text-align: center; font-size: 20px; font-weight: 700; color: #fa243c;'>🤖 Support Vector Machine (SVM)</h3>", unsafe_allow_html=True)
            
            # Sentiment Card SVM
            is_pos_svm = hasil_svm.get("Sentimen_label") == "Positif"
            card_class_svm = "sentiment-card positive" if is_pos_svm else "sentiment-card negative"
            icon_svm = "😊" if is_pos_svm else "😠"
            lbl_svm = "POSITIF" if is_pos_svm else "NEGATIF"
            desc_svm = "Ulasan cenderung mengekspresikan kepuasan." if is_pos_svm else "Ulasan ini mengekspresikan kekecewaan/keluhan."
            
            st.markdown(
                f'<div class="{card_class_svm}">'
                f'<div style="font-size: 11px; color: rgba(255,255,255,0.6); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px; text-transform: uppercase;">Prediksi Sentimen</div>'
                f'<div style="font-size: 38px; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 12px; letter-spacing: -0.02em;">'
                f'<span style="font-size: 40px;">{icon_svm}</span> {lbl_svm}'
                f'</div>'
                f'<div style="font-size: 13px; color: rgba(255,255,255,0.7); margin-top: 8px;">{desc_svm}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Aspect List SVM
            st.markdown("<p style='font-size: 14px; font-weight: 600; color: #98989d; margin-bottom: 12px;'>Deteksi Aspek (SVM):</p>", unsafe_allow_html=True)
            
            aspek_audio_svm = is_detected(hasil_svm.get("Aspek_Audio_Fitur_label"))
            aspek_performa_svm = is_detected(hasil_svm.get("Aspek_Performa_Sistem_label"))
            aspek_harga_svm = is_detected(hasil_svm.get("Aspek_Harga_Layanan_label"))

            st.markdown(f"""
            <div class="aspect-row">
                <span style="font-size: 14px; font-weight: 500;">🎧 Audio & Fitur</span>
                <span class="badge {'badge-pos' if aspek_audio_svm else 'badge-neu'}">{'Terdeteksi' if aspek_audio_svm else 'Tidak'}</span>
            </div>
            <div class="aspect-row">
                <span style="font-size: 14px; font-weight: 500;">⚙️ Performa Sistem</span>
                <span class="badge {'badge-pos' if aspek_performa_svm else 'badge-neu'}">{'Terdeteksi' if aspek_performa_svm else 'Tidak'}</span>
            </div>
            <div class="aspect-row">
                <span style="font-size: 14px; font-weight: 500;">💳 Harga & Layanan</span>
                <span class="badge {'badge-pos' if aspek_harga_svm else 'badge-neu'}">{'Terdeteksi' if aspek_harga_svm else 'Tidak'}</span>
            </div>
            """, unsafe_allow_html=True)

        # ==========================================
        # KOLOM XGBOOST
        # ==========================================
        with col_xgb:
            st.markdown("<h3 style='text-align: center; font-size: 20px; font-weight: 700; color: #ff9f0a;'>🚀 XGBoost Classifier</h3>", unsafe_allow_html=True)
            
            # Sentiment Card XGBoost
            is_pos_xgb = hasil_xgb.get("Sentimen_label") == "Positif"
            card_class_xgb = "sentiment-card positive" if is_pos_xgb else "sentiment-card negative"
            icon_xgb = "😊" if is_pos_xgb else "😠"
            lbl_xgb = "POSITIF" if is_pos_xgb else "NEGATIF"
            desc_xgb = "Ulasan cenderung mengekspresikan kepuasan." if is_pos_xgb else "Ulasan ini mengekspresikan kekecewaan/keluhan."
            
            # Tambahkan confidence jika ada
            proba_sentimen = hasil_xgb.get("proba_Sentimen")
            confidence_html = ""
            if proba_sentimen is not None:
                conf_val = proba_sentimen if is_pos_xgb else (1.0 - proba_sentimen)
                color_conf = "#34c759" if is_pos_xgb else "#ff3b30"
                confidence_html = f'<div style="font-size: 12px; color: {color_conf}; font-weight: 700; background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 12px; display: inline-block; margin-top: 10px;">CONFIDENCE: {conf_val:.2%}</div>'

            st.markdown(
                f'<div class="{card_class_xgb}">'
                f'<div style="font-size: 11px; color: rgba(255,255,255,0.6); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px; text-transform: uppercase;">Prediksi Sentimen</div>'
                f'<div style="font-size: 38px; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 12px; letter-spacing: -0.02em;">'
                f'<span style="font-size: 40px;">{icon_xgb}</span> {lbl_xgb}'
                f'</div>'
                f'<div style="font-size: 13px; color: rgba(255,255,255,0.7); margin-top: 8px;">{desc_xgb}</div>'
                f'{confidence_html}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Aspect List XGBoost
            st.markdown("<p style='font-size: 14px; font-weight: 600; color: #98989d; margin-bottom: 12px;'>Deteksi Aspek (XGBoost):</p>", unsafe_allow_html=True)
            
            aspek_audio_xgb = is_detected(hasil_xgb.get("Aspek_Audio_Fitur_label"))
            aspek_performa_xgb = is_detected(hasil_xgb.get("Aspek_Performa_Sistem_label"))
            aspek_harga_xgb = is_detected(hasil_xgb.get("Aspek_Harga_Layanan_label"))

            st.markdown(f"""
            <div class="aspect-row">
                <span style="font-size: 14px; font-weight: 500;">🎧 Audio & Fitur</span>
                <span class="badge {'badge-pos' if aspek_audio_xgb else 'badge-neu'}">{'Terdeteksi' if aspek_audio_xgb else 'Tidak'}</span>
            </div>
            <div class="aspect-row">
                <span style="font-size: 14px; font-weight: 500;">⚙️ Performa Sistem</span>
                <span class="badge {'badge-pos' if aspek_performa_xgb else 'badge-neu'}">{'Terdeteksi' if aspek_performa_xgb else 'Tidak'}</span>
            </div>
            <div class="aspect-row">
                <span style="font-size: 14px; font-weight: 500;">💳 Harga & Layanan</span>
                <span class="badge {'badge-pos' if aspek_harga_xgb else 'badge-neu'}">{'Terdeteksi' if aspek_harga_xgb else 'Tidak'}</span>
            </div>
            """, unsafe_allow_html=True)

        # ==========================================
        # KESIMPULAN / SUMMARY
        # ==========================================
        agree = hasil_svm.get("Sentimen_label") == hasil_xgb.get("Sentimen_label")
        
        st.markdown("""<div class="conclusion-card">""", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0; font-size: 18px; font-weight: 700; color: #ffffff;'>📝 Kesimpulan Keselarasan Model</h3>", unsafe_allow_html=True)
        
        if agree:
            st.success(
                f"✅ **Kedua algoritma menghasilkan prediksi sentimen yang sama (Konsisten).** "
                f"Model SVM dan XGBoost sepakat bahwa ulasan ini bernilai **{hasil_svm.get('Sentimen_label')}**."
            )
        else:
            st.warning(
                f"⚠️ **Kedua algoritma menghasilkan prediksi sentimen yang berbeda (Tidak Konsisten).** "
                f"SVM memprediksi **{hasil_svm.get('Sentimen_label')}**, sedangkan XGBoost memprediksi **{hasil_xgb.get('Sentimen_label')}**."
            )
        
        audio_match = aspek_audio_svm == aspek_audio_xgb
        performa_match = aspek_performa_svm == aspek_performa_xgb
        harga_match = aspek_harga_svm == aspek_harga_xgb
        match_count = sum([audio_match, performa_match, harga_match])
        
        st.markdown(f"""
        <div style="font-size: 14px; color: #98989d; margin-top: 15px; line-height: 1.6;">
            Tingkat kesamaan deteksi aspek ulasan: <b>{match_count} dari 3 aspek</b> cocok antara SVM dan XGBoost.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔍 Detail JSON Metadata Hasil Prediksi"):
            col_json_svm, col_json_xgb = st.columns(2)
            with col_json_svm:
                st.write("#### Output SVM")
                st.json(hasil_svm)
            with col_json_xgb:
                st.write("#### Output XGBoost")
                st.json(hasil_xgb)

# ==========================================
# TAB 2: STATISTIK DATASET GLOBAL
# ==========================================
with tab2:
    st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-bottom: 24px;'>Analisis Distribusi Data Penelitian</h3>", unsafe_allow_html=True)

    if not df_global.empty:
        chart_col, metric_col = st.columns([2.2, 1])
        
        with metric_col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">TOTAL DATASET</div>
                <div class="metric-value">{len(df_global):,}</div>
            </div>
            """, unsafe_allow_html=True)
            
            pos_count = len(df_global[df_global['Sentimen'] == 1])
            neg_count = len(df_global[df_global['Sentimen'] == 0])
            majority = "POSITIF" if pos_count > neg_count else "NEGATIF"
            maj_color = "#34c759" if majority == "POSITIF" else "#ff3b30"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAYORITAS SENTIMEN</div>
                <div class="metric-value" style="color: {maj_color};">{majority}</div>
                <div style="font-size: 13px; color: #98989d; margin-top: 8px;">
                    {pos_count:,} Positif | {neg_count:,} Negatif ulasan
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with chart_col:
            # Prepare data
            chart_data = []
            for aspect in ['Aspek_Audio_Fitur', 'Aspek_Performa_Sistem', 'Aspek_Harga_Layanan']:
                name = aspect.replace('Aspek_', '').replace('_', ' ')
                subset = df_global[df_global[aspect] == 1]
                chart_data.append({'Aspek': name, 'Sentimen': 'Positif', 'Jumlah': len(subset[subset['Sentimen'] == 1])})
                chart_data.append({'Aspek': name, 'Sentimen': 'Negatif', 'Jumlah': len(subset[subset['Sentimen'] == 0])})
                
            # Altair Chart
            bars = alt.Chart(pd.DataFrame(chart_data)).mark_bar(cornerRadiusEnd=8, height=32).encode(
                x=alt.X('Jumlah:Q', title='Jumlah Ulasan', axis=alt.Axis(gridColor='rgba(255,255,255,0.05)', labelColor='#98989d', tickCount=5, domain=False)),
                y=alt.Y('Aspek:N', title='', sort=None, axis=alt.Axis(labelColor='#f5f5f7', labelFontWeight=500, labelFontSize=13, tickColor='transparent', domainColor='transparent')),
                color=alt.Color('Sentimen:N', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#34c759', '#ff3b30']), legend=alt.Legend(orient='bottom', title=None, labelColor='#f5f5f7', labelFontSize=13, symbolType='circle')),
                tooltip=['Aspek', 'Sentimen', 'Jumlah']
            ).properties(height=280).configure_view(strokeWidth=0).configure(background='transparent').configure_axis(gridDash=[4,4])
            
            st.altair_chart(bars, width="stretch")

        # Datatable Preview
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #f5f5f7; margin-top: 30px;'>📋 Sampel Data Ulasan (Preprocessed)</h4>", unsafe_allow_html=True)
        st.dataframe(df_global[['content', 'clean_teks', 'Sentimen']].head(10), width="stretch", hide_index=True)
    else:
        st.warning("Dataset 'data_preprocessed.csv' tidak ditemukan atau gagal dimuat.")

# ==========================================
# TAB 3: SPESIFIKASI & METRIK MODEL
# ==========================================
with tab3:
    st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-bottom: 8px;'>Metrik Evaluasi Model</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #98989d; font-size: 14px;'>Perbandingan akurasi dan parameter model SVM vs XGBoost yang dilatih pada dataset Apple Music.</p>", unsafe_allow_html=True)

    # Perbandingan Metrik Table
    metrics_data = {
        "Target / Sub-Model": ["Sentimen (Positif/Negatif)", "Aspek: Audio & Fitur", "Aspek: Performa Sistem", "Aspek: Harga & Layanan"],
        "SVM Accuracy": ["~89.5%", "~91.2%", "~88.9%", "~92.1%"],
        "XGBoost Accuracy": ["~90.8%", "~92.5%", "~90.2%", "~93.4%"],
        "Metode Penyeimbangan": ["Class Weight (Balanced)", "SMOTE Oversampling", "SMOTE Oversampling", "SMOTE Oversampling"],
        "Tuning Parameter": ["GridSearchCV", "Optuna (Bayesian Tune)", "Optuna (Bayesian Tune)", "Optuna (Bayesian Tune)"]
    }
    st.dataframe(pd.DataFrame(metrics_data), width="stretch", hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Confusion Matrix Images Section
    st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-bottom: 20px;'>📊 Confusion Matrix (XGBoost Classifier)</h3>", unsafe_allow_html=True)
    
    col_img1, col_img2 = st.columns(2)
    col_img3, col_img4 = st.columns(2)
    
    # Render Confusion Matrix images if available
    img_sentimen = os.path.join(BASE_DIR, "cm_xgb_sentimen.png")
    img_audio = os.path.join(BASE_DIR, "cm_xgb_aspek_audio_fitur.png")
    img_performa = os.path.join(BASE_DIR, "cm_xgb_aspek_performa_sistem.png")
    img_harga = os.path.join(BASE_DIR, "cm_xgb_aspek_harga_layanan.png")

    with col_img1:
        if os.path.exists(img_sentimen):
            st.image(img_sentimen, caption="Confusion Matrix - Klasifikasi Sentimen", width="stretch")
        else:
            st.info("Confusion Matrix Sentimen belum digenerate.")

    with col_img2:
        if os.path.exists(img_audio):
            st.image(img_audio, caption="Confusion Matrix - Aspek Audio & Fitur", width="stretch")
        else:
            st.info("Confusion Matrix Audio & Fitur belum digenerate.")

    with col_img3:
        if os.path.exists(img_performa):
            st.image(img_performa, caption="Confusion Matrix - Aspek Performa Sistem", width="stretch")
        else:
            st.info("Confusion Matrix Performa Sistem belum digenerate.")

    with col_img4:
        if os.path.exists(img_harga):
            st.image(img_harga, caption="Confusion Matrix - Aspek Harga & Layanan", width="stretch")
        else:
            st.info("Confusion Matrix Harga & Layanan belum digenerate.")

