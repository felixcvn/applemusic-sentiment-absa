import streamlit as st
import pandas as pd
import altair as alt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import hasil prediksi probabilitas dari sub-model independen
from xgb_boost import predict_single as predict_xgb
from svm_predict import predict_single as predict_svm

# ==========================================
# Konfigurasi Halaman Dashboard Premium
# ==========================================
st.set_page_config(
    page_title="Dashboard Ensemble Sentimen & ABSA",
    page_icon="🍎",
    layout="wide"
)

# Custom CSS Premium Aesthetic (Luxury Dark Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        background-color: #000000;
    }

    .stApp {
        background-color: #000000;
        color: #f5f5f7;
    }
    
    #MainMenu, header, footer {visibility: hidden;}
    
    h1, h2, h3, h4 {
        letter-spacing: -0.02em !important;
    }

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
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #fa243c !important;
        box-shadow: 0 0 0 1px #fa243c, 0 0 20px rgba(250, 36, 60, 0.15) !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #fa243c 0%, #ff3b30 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(250, 36, 60, 0.35) !important;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 32px rgba(250, 36, 60, 0.5) !important;
    }

    .ensemble-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(28, 28, 30, 0.6);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        gap: 24px;
        transition: all 0.3s ease;
    }
    .ensemble-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.5);
    }
    .ensemble-card.negative {
        background: linear-gradient(135deg, rgba(44, 20, 22, 0.85) 0%, rgba(28, 28, 30, 0.9) 100%);
        border: 1px solid rgba(250, 36, 60, 0.3);
        box-shadow: 0 8px 32px rgba(250, 36, 60, 0.15);
    }
    .ensemble-card.positive {
        background: linear-gradient(135deg, rgba(20, 44, 28, 0.85) 0%, rgba(28, 28, 30, 0.9) 100%);
        border: 1px solid rgba(46, 204, 113, 0.3);
        box-shadow: 0 8px 32px rgba(46, 204, 113, 0.15);
    }
    
    .aspect-widget {
        background: rgba(28, 28, 30, 0.5);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 24px;
        height: 100%;
        transition: all 0.3s ease;
    }
    .aspect-widget:hover {
        transform: translateY(-4px);
        background: rgba(44, 44, 46, 0.7);
    }

    .badge {
        font-size: 11px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 16px;
        text-transform: uppercase;
    }
    .badge-pos { background-color: rgba(52, 199, 89, 0.15); color: #34c759; border: 1px solid rgba(52, 199, 89, 0.2); }
    .badge-neu { background-color: rgba(255, 255, 255, 0.05); color: #98989d; border: 1px solid rgba(255, 255, 255, 0.05); }
    
    .metric-card {
        background: rgba(28, 28, 30, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 16px;
    }
    .metric-title { font-size: 12px; color: #98989d; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 40px; font-weight: 800; color: #ffffff; margin-top: 4px; }
    hr { border-color: rgba(255,255,255,0.08); margin: 40px 0; }
</style>
""", unsafe_allow_html=True)

# --- HEADER TITLE ---
st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
    <div style="background: linear-gradient(135deg, #fa243c 0%, #ff3b30 100%); width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(250,36,60,0.4);">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white" stroke="none">
            <path d="M21 3v12.5a3.5 3.5 0 0 1-3.5 3.5A3.5 3.5 0 0 1 14 15.5a3.5 3.5 0 0 1 3.5-3.5c.54 0 1.05.12 1.5.34V6.47L9 8.6v9.9a3.5 3.5 0 0 1-3.5 3.5A3.5 3.5 0 0 1 2 18.5 3.5 3.5 0 0 1 5.5 15c.54 0 1.05.12 1.5.34V3l14-2v2z"/>
        </svg>
    </div>
    <h1 style='font-weight: 800; font-size: 36px; margin: 0; padding: 0;'>Apple Music Analyzer</h1>
</div>
<p style='color: #98989d; font-size: 16px; margin-left: 64px; font-weight: 500;'>Analisis Sentimen Berbasis Aspek (ABSA) Menggunakan Metode Gabungan <b>Ensemble Learning (SVM + XGBoost)</b></p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- TABS CREATION ---
tab1, tab2 = st.tabs([
    "🔮 Klasifikasi Sistem Ensemble", 
    "📊 Statistik Dataset Global"
])

# Load Dataset Global
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(os.path.join(BASE_DIR, 'data_preprocessed.csv')).dropna(subset=['clean_teks'])
    except:
        return pd.DataFrame()

df_global = load_dataset()

# ==========================================
# TAB 1: REAL-TIME ENSEMBLE KEPUTUSAN TUNGGAL
# ==========================================
with tab1:
    col_input, col_space, col_output = st.columns([1.2, 0.1, 1.5])

    with col_input:
        st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-top: 0; margin-bottom: 20px;'>Uji Sentimen Real-time</h3>", unsafe_allow_html=True)
        user_input = st.text_area("Masukkan ulasan:", placeholder="Ketik ulasan di sini (misal: Audio dolby jernih banget tapi aplikasinya sering ngelag...)", height=175, label_visibility="collapsed")
        analyze_btn = st.button("Jalankan Prediksi Ensemble", width="stretch")

    with col_output:
        st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-top: 0; margin-bottom: 20px;'>Hasil Analisis Sentimen Ensemble</h3>", unsafe_allow_html=True)
        
        if not analyze_btn:
            st.markdown("""
            <div style="background: rgba(28, 28, 30, 0.3); border: 2px dashed rgba(255,255,255,0.08); border-radius: 24px; padding: 40px; text-align: center; color: #636366; min-height: 240px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="font-size: 40px; margin-bottom: 16px; opacity: 0.5;">🤖</div>
                <div style="font-size: 16px; font-weight: 600; color: #98989d;">Menunggu input ulasan proyek.</div>
                <div style="font-size: 13px; margin-top: 8px;">Sistem Ensemble akan menggabungkan prediksi dari model SVM dan XGBoost.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if not user_input.strip():
                st.warning("Teks ulasan tidak boleh kosong.")
            else:
                with st.spinner("Mengkalkulasi Probabilitas Soft-Voting Ensemble..."):
                    # Ekstrak dari kedua model script
                    hasil_svm = predict_svm(user_input)
                    hasil_xgb = predict_xgb(user_input)

                if "error" in hasil_svm or "error" in hasil_xgb:
                    st.error("Terjadi kegagalan pemrosesan pada salah satu sub-model.")
                else:
                    # -----------------------------------------------------------------
                    # RUMUS EKSEKUSI SOFT VOTING ENSEMBLE (MENGGABUNGKAN PROBABILITAS)
                    # -----------------------------------------------------------------
                    p_sentimen_svm = hasil_svm.get("proba_Sentimen", 1.0 if hasil_svm.get("Sentimen_label") == "Positif" else 0.0)
                    p_sentimen_xgb = hasil_xgb.get("proba_Sentimen", 0.5) # Dari file xgb_boost.py temanmu
                    ensemble_proba_sentimen = (p_sentimen_svm + p_sentimen_xgb) / 2

                    is_pos_final = ensemble_proba_sentimen >= 0.5
                    final_label = "POSITIF" if is_pos_final else "NEGATIF"
                    final_confidence = ensemble_proba_sentimen if is_pos_final else (1.0 - ensemble_proba_sentimen)
                    
                    # Logika Soft-Voting Gabungan untuk Aspek
                    def compute_ensemble_aspect(key):
                        # Ambil nilai probabilitas aspek dari kedua model, jika SVM tidak punya fallback ke 1/0
                        p_svm = hasil_svm.get(f"proba_{key}", 1.0 if hasil_svm.get(f"{key}_label") == "Ya" else 0.0)
                        p_xgb = hasil_xgb.get(f"proba_{key}", 0.0)
                        return ((p_svm + p_xgb) / 2) >= 0.5

                    aspek_audio_final = compute_ensemble_aspect("Aspek_Audio_Fitur")
                    aspek_performa_final = compute_ensemble_aspect("Aspek_Performa_Sistem")
                    aspek_harga_final = compute_ensemble_aspect("Aspek_Harga_Layanan")

                    # RENDER SENTIMEN CARD TUNGGAL (HASIL ENSEMBLE)
                    c_class = "ensemble-card positive" if is_pos_final else "ensemble-card negative"
                    icon = "😄" if is_pos_final else "😠"
                    t_color = "#34c759" if is_pos_final else "#ff3b30"
                    quote = "Konsensus Ensemble: Ulasan mengekspresikan kepuasan." if is_pos_final else "Konsensus Ensemble: Ulasan ini berisi kekecewaan/kritik sistem."

                    st.markdown(f"""
                    <div class="{c_class}">
                        <div>
                            <div style="font-size: 12px; color: rgba(255,255,255,0.6); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px; text-transform: uppercase;">Keputusan Model Ensemble</div>
                            <div style="font-size: 44px; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 16px; letter-spacing: -0.02em;">
                                <span style="font-size: 48px;">{icon}</span> {final_label}
                            </div>
                        </div>
                        <div style="text-align: right; max-width: 50%;">
                            <div style="font-size: 14px; color: rgba(255,255,255,0.8); margin-bottom: 12px; line-height: 1.5;">{quote}</div>
                            <div style="font-size: 12px; color: {t_color}; font-weight: 700; background: rgba(0,0,0,0.3); padding: 5px 12px; border-radius: 12px; display: inline-block;">COMBINED CONFIDENCE: {final_confidence:.2%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # RENDER ASPEK BADGE FULL WIDTH (DI BAWAH KARTU KEPUTUSAN)
    if analyze_btn and user_input.strip() and "error" not in hasil_svm:
        st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #f5f5f7; margin-top: 40px; margin-bottom: 24px;'>Aspek Terintegrasi Hasil Voting</h4>", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        
        def render_aspect_widget(title, is_det, icon, desc_pos, desc_neg):
            b_class = "badge-pos" if is_det else "badge-neu"
            b_text = "Terdeteksi" if is_det else "Tidak Terdeteksi"
            desc = desc_pos if is_det else desc_neg
            opac = "1.0" if is_det else "0.5"
            return f"""
            <div class="aspect-widget" style="opacity: {opac};">
                <div class="badge {b_class}">{b_text}</div>
                <div style="font-size: 28px; margin-bottom: 12px;">{icon}</div>
                <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">{title}</div>
                <div style="font-size: 13px; color: #98989d; line-height: 1.5;">{desc}</div>
            </div>
            """
        with a1: st.markdown(render_aspect_widget("Audio & Fitur", aspek_audio_final, "🎧", "Kualitas audio atau kelengkapan fitur disinggung dalam teks.", "Tidak menyinggung terkait kualitas suara atau fitur."), unsafe_allow_html=True)
        with a2: st.markdown(render_aspect_widget("Performa Sistem", aspek_performa_final, "⚙️", "Keluhan tentang bug, lag, atau kestabilan sistem ditemukan.", "Tidak ada masalah performa yang disebutkan."), unsafe_allow_html=True)
        with a3: st.markdown(render_aspect_widget("Harga Layanan", aspek_harga_final, "💳", "Topik seputar tagihan, biaya, atau pembayaran terdeteksi.", "Tidak menyebutkan soal harga atau biaya langganan."), unsafe_allow_html=True)

        # EXPANDER DETIL INTERNAL UNTUK KEPERLUAN SIDANG/EVALUASI
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Log Rincian Kontribusi Bobot Sub-Model (SVM vs XGBoost)"):
            c_svm, c_xgb = st.columns(2)
            with c_svm:
                st.write("**Metrik Prediksi SVM (Kamu):**")
                st.json(hasil_svm)
            with c_xgb:
                st.write("**Metrik Prediksi XGBoost (Temanmu):**")
                st.json(hasil_xgb)

# ==========================================
# TAB 2 & TAB 3 (Menggunakan visualisasi data dari file asli kamu)
# ==========================================
with tab2:
    st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-bottom: 32px;'>Statistik Dataset Global</h3>", unsafe_allow_html=True)
    if not df_global.empty:
        chart_col, metric_col = st.columns([2.5, 1])
        with metric_col:
            st.markdown(f'<div class="metric-card"><div class="metric-title">TOTAL ULASAN</div><div class="metric-value">{len(df_global)}</div></div>', unsafe_allow_html=True)
            pos_c = len(df_global[df_global['Sentimen'] == 1])
            st.markdown(f'<div class="metric-card"><div class="metric-title">MAYORITAS SENTIMEN</div><div class="metric-value" style="color: #34c759;">%s</div><div style="font-size: 13px; color: #98989d; margin-top: 8px;">{pos_c} ulasan positif</div></div>' % "POSITIF", unsafe_allow_html=True)
        with chart_col:
            chart_data = []
            for aspect in ['Aspek_Audio_Fitur', 'Aspek_Performa_Sistem', 'Aspek_Harga_Layanan']:
                name = aspect.replace('Aspek_', '').replace('_', ' ')
                subset = df_global[df_global[aspect] == 1]
                chart_data.append({'Aspek': name, 'Sentimen': 'Positif', 'Jumlah': len(subset[subset['Sentimen'] == 1])})
                chart_data.append({'Aspek': name, 'Sentimen': 'Negatif', 'Jumlah': len(subset[subset['Sentimen'] == 0])})
            bars = alt.Chart(pd.DataFrame(chart_data)).mark_bar(cornerRadiusEnd=8, height=32).encode(
                x=alt.X('Jumlah:Q', title=''), y=alt.Y('Aspek:N', title='', sort=None),
                color=alt.Color('Sentimen:N', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#34c759', '#ff3b30'])),
                tooltip=['Aspek', 'Sentimen', 'Jumlah']
            ).properties(height=280).configure_view(strokeWidth=0).configure(background='transparent')
            st.altair_chart(bars, width="stretch")