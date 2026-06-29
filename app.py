import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import plotly.graph_objects as go
import plotly.express as px
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="Fuzzy Discount Optimizer | Olist E-commerce",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom Styling (Emerald & Slate Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    .main-title {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* Card container styling */
    .metric-card {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.4);
    }
    
    .metric-title {
        font-size: 0.875rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #111827;
        margin-top: 0.25rem;
    }
    
    /* For dark mode adaptability */
    @media (prefers-color-scheme: dark) {
        .metric-value {
            color: #F9FAFB;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.08);
        }
        .metric-card:hover {
            border-color: rgba(16, 185, 129, 0.3);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #10B981;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Path Configurations & Data Loading ---
DATA_DIR = "dataset"
RESULTS_FILE = os.path.join(DATA_DIR, "fis_discount_results.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "product_features_clean.csv")

@st.cache_data
def load_data():
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(CLEAN_FILE):
        return None, None
    df_results = pd.read_csv(RESULTS_FILE)
    df_clean = pd.read_csv(CLEAN_FILE)
    return df_results, df_clean

df_results, df_clean = load_data()

# Fallback in case datasets are missing
if df_clean is None:
    st.error("❌ Dataset tidak ditemukan. Harap jalankan notebook pemrosesan data terlebih dahulu untuk menghasilkan file di folder `dataset/`.")
    st.info("💡 Jalankan: `python generate_notebooks.py` lalu jalankan semua sel di notebook 2 & 3 untuk membuat dataset.")
    st.stop()

# --- Precompute Percentiles for Membership Functions Calibration ---
# We calculate statistics dynamically based on the clean dataset to ensure flexibility
stats = df_clean[['stok_level_log', 'jumlah_penjualan_log', 'stok_level_rank', 'jumlah_penjualan_rank', 'avg_rating']].describe(percentiles=[.25, .33, .50, .67, .75])

# Dynamic Percentiles for Rank-Based
MFs_stok_rank = {
    'P25': stats.loc['25%', 'stok_level_rank'],
    'P33': stats.loc['33%', 'stok_level_rank'],
    'P50': stats.loc['50%', 'stok_level_rank'],
    'P67': stats.loc['67%', 'stok_level_rank'],
    'P75': stats.loc['75%', 'stok_level_rank']
}
MFs_penjualan_rank = {
    'P25': stats.loc['25%', 'jumlah_penjualan_rank'],
    'P33': stats.loc['33%', 'jumlah_penjualan_rank'],
    'P50': stats.loc['50%', 'jumlah_penjualan_rank'],
    'P67': stats.loc['67%', 'jumlah_penjualan_rank'],
    'P75': stats.loc['75%', 'jumlah_penjualan_rank']
}

# Dynamic Percentiles for Log-Transform
MFs_stok_log = {
    'P25': stats.loc['25%', 'stok_level_log'],
    'P33': stats.loc['33%', 'stok_level_log'],
    'P50': stats.loc['50%', 'stok_level_log'],
    'P67': stats.loc['67%', 'stok_level_log'],
    'P75': stats.loc['75%', 'stok_level_log']
}
MFs_penjualan_log = {
    'P25': stats.loc['25%', 'jumlah_penjualan_log'],
    'P33': stats.loc['33%', 'jumlah_penjualan_log'],
    'P50': stats.loc['50%', 'jumlah_penjualan_log'],
    'P67': stats.loc['67%', 'jumlah_penjualan_log'],
    'P75': stats.loc['75%', 'jumlah_penjualan_log']
}

P50_rating = stats.loc['50%', 'avg_rating']

# --- Cache Fuzzy Control Systems ---
@st.cache_resource
def get_fuzzy_simulation(model_type):
    # Define Antecedents and Consequent
    stok = ctrl.Antecedent(np.arange(0, 101, 1), 'stok_level')
    penjualan = ctrl.Antecedent(np.arange(0, 101, 1), 'jumlah_penjualan')
    rating = ctrl.Antecedent(np.arange(1.0, 5.1, 0.1), 'avg_rating')
    diskon = ctrl.Consequent(np.arange(0, 51, 1), 'besar_diskon')

    # Define Consequent Output Membership Functions (Same for all models)
    diskon['Kecil'] = fuzz.trimf(diskon.universe, [0, 0, 20])
    diskon['Sedang'] = fuzz.trimf(diskon.universe, [10, 25, 40])
    diskon['Besar'] = fuzz.trimf(diskon.universe, [30, 50, 50])

    if model_type == 'sebelum':
        # Heuristic Configuration
        stok['Sedikit'] = fuzz.trimf(stok.universe, [0, 0, 40])
        stok['Sedang'] = fuzz.trimf(stok.universe, [20, 50, 80])
        stok['Banyak'] = fuzz.trimf(stok.universe, [60, 100, 100])

        penjualan['Rendah'] = fuzz.trimf(penjualan.universe, [0, 0, 40])
        penjualan['Sedang'] = fuzz.trimf(penjualan.universe, [20, 50, 80])
        penjualan['Tinggi'] = fuzz.trimf(penjualan.universe, [60, 100, 100])

        rating['Buruk'] = fuzz.trimf(rating.universe, [1.0, 1.0, 3.0])
        rating['Baik'] = fuzz.trimf(rating.universe, [3.0, 5.0, 5.0])
        
    elif model_type == 'log':
        # Calibrated Log-Transform configuration
        stok['Sedikit'] = fuzz.trimf(stok.universe, [0, 0, MFs_stok_log['P33']])
        stok['Sedang'] = fuzz.trimf(stok.universe, [MFs_stok_log['P25'], MFs_stok_log['P50'], MFs_stok_log['P75']])
        stok['Banyak'] = fuzz.trimf(stok.universe, [MFs_stok_log['P67'], 100, 100])

        penjualan['Rendah'] = fuzz.trimf(penjualan.universe, [0, 0, MFs_penjualan_log['P33']])
        penjualan['Sedang'] = fuzz.trimf(penjualan.universe, [MFs_penjualan_log['P25'], MFs_penjualan_log['P50'], MFs_penjualan_log['P75']])
        penjualan['Tinggi'] = fuzz.trimf(penjualan.universe, [MFs_penjualan_log['P67'], 100, 100])

        rating['Buruk'] = fuzz.trimf(rating.universe, [1.0, 1.0, P50_rating])
        rating['Baik'] = fuzz.trimf(rating.universe, [P50_rating, 5.0, 5.0])
        
    else:  # 'rank'
        # Calibrated Rank-Based configuration
        stok['Sedikit'] = fuzz.trimf(stok.universe, [0, 0, MFs_stok_rank['P33']])
        stok['Sedang'] = fuzz.trimf(stok.universe, [MFs_stok_rank['P25'], MFs_stok_rank['P50'], MFs_stok_rank['P75']])
        stok['Banyak'] = fuzz.trimf(stok.universe, [MFs_stok_rank['P67'], 100, 100])

        penjualan['Rendah'] = fuzz.trimf(penjualan.universe, [0, 0, MFs_penjualan_rank['P33']])
        penjualan['Sedang'] = fuzz.trimf(penjualan.universe, [MFs_penjualan_rank['P25'], MFs_penjualan_rank['P50'], MFs_penjualan_rank['P75']])
        penjualan['Tinggi'] = fuzz.trimf(penjualan.universe, [MFs_penjualan_rank['P67'], 100, 100])

        rating['Buruk'] = fuzz.trimf(rating.universe, [1.0, 1.0, P50_rating])
        rating['Baik'] = fuzz.trimf(rating.universe, [P50_rating, 5.0, 5.0])

    # Rule Base (8 Rules)
    rule1 = ctrl.Rule(stok['Banyak'] & penjualan['Rendah'], diskon['Besar'])
    rule2 = ctrl.Rule(stok['Sedikit'] & penjualan['Tinggi'], diskon['Kecil'])
    rule3 = ctrl.Rule(rating['Buruk'], diskon['Besar'])
    rule4 = ctrl.Rule(stok['Sedang'] & penjualan['Sedang'], diskon['Sedang'])
    rule5 = ctrl.Rule(stok['Banyak'] & penjualan['Sedang'], diskon['Sedang'])
    rule6 = ctrl.Rule(stok['Sedang'] & penjualan['Rendah'], diskon['Sedang'])
    rule7 = ctrl.Rule(stok['Sedikit'] & penjualan['Sedang'], diskon['Kecil'])
    rule8 = ctrl.Rule(rating['Baik'] & penjualan['Tinggi'], diskon['Kecil'])

    discount_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
    discount_sim = ctrl.ControlSystemSimulation(discount_ctrl)
    
    return discount_sim, (stok, penjualan, rating, diskon)

# --- Helper Functions ---
def get_discount_tier(d):
    if d <= 17.0:
        return 'Kecil'
    elif d <= 33.0:
        return 'Sedang'
    else:
        return 'Besar'

def map_raw_to_fuzzy_inputs(raw_sales, raw_rating, df):
    """Maps raw product input into respective fuzzy universes based on empirical distributions."""
    # Rank-Based Calculation
    sales_rank = (df['jumlah_penjualan'] < raw_sales).sum() / len(df) * 100.0
    equal_count = (df['jumlah_penjualan'] == raw_sales).sum()
    if equal_count > 1:
        sales_rank += (equal_count - 1) / 2.0 / len(df) * 100.0
    stok_rank = 100.0 - sales_rank
    
    # Log-Transform Calculation
    log_sales_raw = np.log1p(raw_sales)
    min_log = np.log1p(df['total_sold']).min()
    max_log = np.log1p(df['total_sold']).max()
    sales_log = (log_sales_raw - min_log) / (max_log - min_log) * 100.0
    sales_log = np.clip(sales_log, 0.0, 100.0)
    stok_log = 100.0 - sales_log
    
    # Heuristic / Original Min-Max Calculation
    min_jual = df['jumlah_penjualan'].min()
    max_jual = df['jumlah_penjualan'].max()
    sales_orig = (raw_sales - min_jual) / (max_jual - min_jual) * 100.0
    sales_orig = np.clip(sales_orig, 0.0, 100.0)
    
    min_sold = df['total_sold'].min()
    max_sold = df['total_sold'].max()
    stok_orig = 100.0 - ((raw_sales - min_sold) / (max_sold - min_sold) * 100.0)
    stok_orig = np.clip(stok_orig, 0.0, 100.0)
    
    return {
        'rank': (stok_rank, sales_rank, raw_rating),
        'log': (stok_log, sales_log, raw_rating),
        'orig': (stok_orig, sales_orig, raw_rating)
    }

def get_membership_activations(stok_val, penjualan_val, rating_val, variables):
    """Calculates activation degree of membership functions for visual explanation."""
    stok, penjualan, rating, diskon = variables
    stok_acts = {
        'Sedikit': float(fuzz.interp_membership(stok.universe, stok['Sedikit'].mf, stok_val)),
        'Sedang': float(fuzz.interp_membership(stok.universe, stok['Sedang'].mf, stok_val)),
        'Banyak': float(fuzz.interp_membership(stok.universe, stok['Banyak'].mf, stok_val))
    }
    penjualan_acts = {
        'Rendah': float(fuzz.interp_membership(penjualan.universe, penjualan['Rendah'].mf, penjualan_val)),
        'Sedang': float(fuzz.interp_membership(penjualan.universe, penjualan['Sedang'].mf, penjualan_val)),
        'Tinggi': float(fuzz.interp_membership(penjualan.universe, penjualan['Tinggi'].mf, penjualan_val))
    }
    rating_acts = {
        'Buruk': float(fuzz.interp_membership(rating.universe, rating['Buruk'].mf, rating_val)),
        'Baik': float(fuzz.interp_membership(rating.universe, rating['Baik'].mf, rating_val))
    }
    return stok_acts, penjualan_acts, rating_acts

def get_rule_activations(stok_acts, penjualan_acts, rating_acts):
    """Computes final firing strength for the 8 Mamdani rules."""
    rules = [
        {"Rule": "Rule 1", "Deskripsi": "IF Stok Banyak & Penjualan Rendah THEN Diskon Besar", "val": min(stok_acts['Banyak'], penjualan_acts['Rendah'])},
        {"Rule": "Rule 2", "Deskripsi": "IF Stok Sedikit & Penjualan Tinggi THEN Diskon Kecil", "val": min(stok_acts['Sedikit'], penjualan_acts['Tinggi'])},
        {"Rule": "Rule 3", "Deskripsi": "IF Rating Buruk THEN Diskon Besar", "val": rating_acts['Buruk']},
        {"Rule": "Rule 4", "Deskripsi": "IF Stok Sedang & Penjualan Sedang THEN Diskon Sedang", "val": min(stok_acts['Sedang'], penjualan_acts['Sedang'])},
        {"Rule": "Rule 5", "Deskripsi": "IF Stok Banyak & Penjualan Sedang THEN Diskon Sedang", "val": min(stok_acts['Banyak'], penjualan_acts['Sedang'])},
        {"Rule": "Rule 6", "Deskripsi": "IF Stok Sedang & Penjualan Rendah THEN Diskon Sedang", "val": min(stok_acts['Sedang'], penjualan_acts['Rendah'])},
        {"Rule": "Rule 7", "Deskripsi": "IF Stok Sedikit & Penjualan Sedang THEN Diskon Kecil", "val": min(stok_acts['Sedikit'], penjualan_acts['Sedang'])},
        {"Rule": "Rule 8", "Deskripsi": "IF Rating Baik & Penjualan Tinggi THEN Diskon Kecil", "val": min(rating_acts['Baik'], penjualan_acts['Tinggi'])},
    ]
    return pd.DataFrame(rules)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ <span>Fuzzy Optimizer</span></div>', unsafe_allow_html=True)
    st.markdown("### Menu Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        ["📈 Ringkasan & Data Explorer", "🎛️ Simulator Logika Fuzzy", "📊 Visualisasi Analisis", "📚 Aturan & Metodologi"]
    )
    
    st.markdown("---")
    st.markdown("### Metadata Sistem")
    st.info("""
    **Studi Kasus:** Olist Brasil  
    **Metode:** Mamdani FIS  
    **Output:** Persentase Diskon (0% - 50%)  
    """)

# ==============================================================================
# PAGE 1: OVERVIEW & DATA EXPLORER
# ==============================================================================
if page == "📈 Ringkasan & Data Explorer":
    st.markdown('<h1 class="main-title">Ringkasan Sistem & Eksplorasi Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Visualisasi performa sistem penentuan diskon fuzzy dan data ritel Olist Brasil.</p>', unsafe_allow_html=True)
    
    # Overview Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Produk Aktif</div>
            <div class="metric-value">{len(df_results):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rata-rata Rating</div>
            <div class="metric-value">⭐️ {df_results['avg_rating'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rata-rata Diskon (Rank)</div>
            <div class="metric-value">{df_results['besar_diskon_rank'].mean():.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rata-rata Diskon (Sebelum)</div>
            <div class="metric-value">{df_results['besar_diskon_sebelum'].mean():.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🔍 Eksplorasi & Filter Katalog Produk")
    
    # Filter Row
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        search_query = st.text_input("Cari berdasarkan Product ID:", placeholder="Masukkan hash ID produk...")
    with f2:
        categories = sorted(df_results['category'].unique())
        selected_categories = st.multiselect("Filter Kategori Produk:", categories)
    with f3:
        tier_filter = st.multiselect("Tier Diskon (Rank):", ["Kecil", "Sedang", "Besar"], default=["Kecil", "Sedang", "Besar"])
        
    # Apply filters
    filtered_df = df_results.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df['product_id'].str.contains(search_query, case=False)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
        
    filtered_df['Tier_Rank'] = filtered_df['besar_diskon_rank'].apply(get_discount_tier)
    filtered_df = filtered_df[filtered_df['Tier_Rank'].isin(tier_filter)]
    
    # Clean display
    display_df = filtered_df[[
        'product_id', 'category', 'jumlah_penjualan', 'total_sold', 'avg_rating', 
        'stok_level_rank', 'besar_diskon_sebelum', 'besar_diskon_rank'
    ]].copy()
    display_df.columns = [
        'Product ID', 'Kategori', 'Unique Orders', 'Total Units Sold', 'Avg Rating',
        'Stock Level (Rank)', 'Diskon Heuristik (%)', 'Diskon Calibrated Rank (%)'
    ]
    
    st.write(f"Menampilkan **{len(display_df):,}** produk dari total **{len(df_results):,}**.")
    st.dataframe(display_df.style.format({
        'Avg Rating': '{:.2f}',
        'Stock Level (Rank)': '{:.1f}',
        'Diskon Heuristik (%)': '{:.2f}%',
        'Diskon Calibrated Rank (%)': '{:.2f}%'
    }), use_container_width=True, height=400)
    
    # Downloader
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Hasil Filter sebagai CSV",
        data=csv_data,
        file_name="fuzzy_discount_filtered.csv",
        mime="text/csv"
    )

# ==============================================================================
# PAGE 2: FUZZY SIMULATOR
# ==============================================================================
elif page == "🎛️ Simulator Logika Fuzzy":
    st.markdown('<h1 class="main-title">Simulator Inferensi Fuzzy Dinamis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Sesuaikan parameter produk untuk menghitung rekomendasi diskon secara real-time.</p>', unsafe_allow_html=True)
    
    # Session state for product ID input helper
    if "input_sales" not in st.session_state:
        st.session_state["input_sales"] = 15
    if "input_rating" not in st.session_state:
        st.session_state["input_rating"] = 4.2
    if "input_stock_pct" not in st.session_state:
        st.session_state["input_stock_pct"] = 50.0
    if "input_sales_pct" not in st.session_state:
        st.session_state["input_sales_pct"] = 50.0

    # Quick Load product from DB
    st.markdown("### 📥 Isi Otomatis Menggunakan Produk Rill")
    search_prod_id = st.text_input("Cari Product ID dari database untuk disalin:", placeholder="Masukkan hash ID produk...")
    if search_prod_id:
        match = df_results[df_results['product_id'] == search_prod_id]
        if not match.empty:
            p_data = match.iloc[0]
            st.success(f"✅ Produk ditemukan! Kategori: `{p_data['category']}`")
            # Save into session state
            st.session_state["input_sales"] = int(p_data['jumlah_penjualan'])
            st.session_state["input_rating"] = float(p_data['avg_rating'])
            st.session_state["input_stock_pct"] = float(p_data['stok_level_rank'])
            st.session_state["input_sales_pct"] = float(p_data['jumlah_penjualan_rank'])
        else:
            st.warning("⚠️ Product ID tidak ditemukan di database.")

    # Two column layout: Control Panel & Calculations
    c_panel, c_results = st.columns([1, 2])
    
    with c_panel:
        st.markdown("### 🛠️ Panel Parameter")
        
        input_mode = st.radio("Pilih Mode Input:", ["Nilai Riil Produk", "Persentase Semesta Himpunan (0-100)"])
        
        model_selection = st.selectbox(
            "Pilih Varian Model FIS:",
            ["Rank-Based (Calibrated)", "Log-Transform (Calibrated)", "Sebelum (Heuristic)"]
        )
        
        # Mapping model types
        model_map = {
            "Rank-Based (Calibrated)": "rank",
            "Log-Transform (Calibrated)": "log",
            "Sebelum (Heuristic)": "sebelum"
        }
        active_model = model_map[model_selection]
        
        if input_mode == "Nilai Riil Produk":
            raw_sales = st.slider("Jumlah Penjualan (Unique Orders):", 1, 100, st.session_state["input_sales"])
            raw_rating = st.slider("Rata-rata Rating (Review):", 1.0, 5.0, st.session_state["input_rating"], step=0.1)
            
            # Map raw inputs
            mapped = map_raw_to_fuzzy_inputs(raw_sales, raw_rating, df_clean)
            stok_val, penjualan_val, rating_val = mapped[active_model]
            
            # Show how it is mapped
            st.markdown("##### 📍 Nilai Hasil Transformasi (Semesta Fuzzy):")
            st.caption(f"**Stock Level:** `{stok_val:.2f}%` | **Jumlah Penjualan:** `{penjualan_val:.2f}%` | **Rating:** `{rating_val:.1f}`")
            
        else:
            stok_val = st.slider("Stock Level (Persentase):", 0.0, 100.0, st.session_state["input_stock_pct"], step=0.5)
            penjualan_val = st.slider("Sales Volume (Persentase):", 0.0, 100.0, st.session_state["input_sales_pct"], step=0.5)
            rating_val = st.slider("Average Rating (Bintang 1-5):", 1.0, 5.0, st.session_state["input_rating"], step=0.1)
            
        # Get simulation
        sim_obj, vars_setup = get_fuzzy_simulation(active_model)
        
        # Inject inputs
        sim_obj.input['stok_level'] = stok_val
        sim_obj.input['jumlah_penjualan'] = penjualan_val
        sim_obj.input['avg_rating'] = rating_val
        
        # Compute Output
        try:
            sim_obj.compute()
            crisp_output = sim_obj.output['besar_diskon']
            error_status = None
        except Exception as e:
            crisp_output = 25.0 # fallback
            error_status = str(e)
            
    with c_results:
        st.markdown("### 🏆 Hasil Keputusan Inferensi")
        
        if error_status:
            st.warning(f"⚠️ Defuzzifikasi Gagal: {error_status}. Menampilkan nilai default fallback.")
            
        # Large Card output
        tier = get_discount_tier(crisp_output)
        
        # Gauge Plotly Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = crisp_output,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Rekomendasi Diskon (Tier: {tier})", 'font': {'size': 20, 'family': 'Outfit'}},
            gauge = {
                'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#10B981"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 17.0], 'color': '#ECFDF5'},
                    {'range': [17.0, 33.0], 'color': '#D1FAE5'},
                    {'range': [33.0, 50.0], 'color': '#A7F3D0'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': crisp_output
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Side-by-Side Model Comparison
        st.markdown("#### 🔄 Perbandingan Rekomendasi Antar Varian FIS")
        
        # Let's run calculations for the other two models
        comp_results = {}
        for m_name, m_key in [("Sebelum (Heuristik)", "orig"), ("Log-Transform", "log"), ("Rank-Based", "rank")]:
            # If input is percentage, we use raw percentages directly for all models
            # If input is real product values, we map them for each model
            if input_mode == "Nilai Riil Produk":
                s_v, p_v, r_v = map_raw_to_fuzzy_inputs(raw_sales, raw_rating, df_clean)[m_key]
            else:
                s_v, p_v, r_v = stok_val, penjualan_val, rating_val
                
            model_sim, _ = get_fuzzy_simulation(m_name.lower().split()[0].replace("-", ""))
            model_sim.input['stok_level'] = s_v
            model_sim.input['jumlah_penjualan'] = p_v
            model_sim.input['avg_rating'] = r_v
            try:
                model_sim.compute()
                val = model_sim.output['besar_diskon']
            except:
                val = 25.0
            comp_results[m_name] = val
            
        cols_comp = st.columns(3)
        for i, (m_name, val) in enumerate(comp_results.items()):
            highlight = "👈 (Aktif)" if m_name.startswith(model_selection.split()[0]) else ""
            with cols_comp[i]:
                st.metric(label=f"{m_name} {highlight}", value=f"{val:.2f}%", delta=f"{val-comp_results['Sebelum (Heuristik)']:.2f}% vs Heuristik")
                
    st.markdown("---")
    st.markdown("### 🧩 Aktivasi Fungsi Keanggotaan & Aturan (Explainable AI)")
    
    # Calculate activations
    stok_acts, penjualan_acts, rating_acts = get_membership_activations(stok_val, penjualan_val, rating_val, vars_setup)
    
    # Chart for Membership Activation levels
    c_m1, c_m2 = st.columns([1, 1])
    
    with c_m1:
        st.markdown("#### Derajat Keanggotaan Variabel Input")
        
        # Structure data
        act_data = []
        for term, val in stok_acts.items():
            act_data.append({"Variabel": "Stock Level", "Himpunan": term, "Derajat Keanggotaan (μ)": val})
        for term, val in penjualan_acts.items():
            act_data.append({"Variabel": "Jumlah Penjualan", "Himpunan": term, "Derajat Keanggotaan (μ)": val})
        for term, val in rating_acts.items():
            act_data.append({"Variabel": "Average Rating", "Himpunan": term, "Derajat Keanggotaan (μ)": val})
            
        act_df = pd.DataFrame(act_data)
        fig_act = px.bar(
            act_df, 
            x="Himpunan", 
            y="Derajat Keanggotaan (μ)", 
            color="Variabel",
            barmode="group",
            color_discrete_sequence=['#10B981', '#3B82F6', '#8B5CF6'],
            range_y=[0, 1.05]
        )
        fig_act.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_act, use_container_width=True)
        
    with c_m2:
        st.markdown("#### Aturan yang Firing (Firing Strength > 0)")
        rule_df = get_rule_activations(stok_acts, penjualan_acts, rating_acts)
        firing_rules = rule_df[rule_df['val'] > 0].sort_values(by="val", ascending=False)
        
        if firing_rules.empty:
            st.caption("Tidak ada aturan fuzzy yang teraktivasi (derajat keanggotaan 0).")
        else:
            fig_rules = px.bar(
                firing_rules, 
                x="val", 
                y="Rule", 
                orientation='h',
                hover_data=["Deskripsi"],
                labels={"val": "Kekuatan Aktivasi (Firing Strength)", "Rule": "Aturan"},
                color_discrete_sequence=['#059669'],
                range_x=[0, 1.05]
            )
            fig_rules.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_rules, use_container_width=True)
            
            # Text explanation
            st.markdown("**Deskripsi Aturan Aktif Teratas:**")
            top_rule = firing_rules.iloc[0]
            st.info(f"⚡ **{top_rule['Rule']}:** *{top_rule['Deskripsi']}* dengan kekuatan **{top_rule['val']:.2f}**.")

# ==============================================================================
# PAGE 3: ANALYTICAL VISUALIZATIONS
# ==============================================================================
elif page == "📊 Visualisasi Analisis":
    st.markdown('<h1 class="main-title">Visualisasi Kontrol & Evaluasi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pemetaan output non-linear melalui control surface 3D dan hasil evaluasi performa model.</p>', unsafe_allow_html=True)
    
    tab_surface, tab_pregenerated = st.tabs(["🎮 Permukaan Kontrol 3D Interaktif", "📁 Gambar Hasil Evaluasi Laporan"])
    
    with tab_surface:
        st.markdown("### Permukaan Kontrol 3D Dinamis (Interactive Control Surface)")
        st.write("Visualisasikan bagaimana rekomendasi diskon berubah secara dinamis berdasarkan hubungan non-linear tingkat stok dan volume penjualan.")
        
        surf_model = st.selectbox(
            "Pilih Model untuk Control Surface:",
            ["Rank-Based (Calibrated)", "Sebelum (Heuristic)"]
        )
        surf_key = "rank" if surf_model.startswith("Rank") else "sebelum"
        
        # Dynamic Plotly 3D Surface Generator
        @st.cache_data
        def build_interactive_surface(model_type):
            sim, _ = get_fuzzy_simulation(model_type)
            x_range = np.linspace(0, 100, 30)
            y_range = np.linspace(0, 100, 30)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            z_grid = np.zeros_like(x_grid)

            for i in range(x_grid.shape[0]):
                for j in range(x_grid.shape[1]):
                    sim.input['stok_level'] = x_grid[i, j]
                    sim.input['jumlah_penjualan'] = y_grid[i, j]
                    sim.input['avg_rating'] = 4.0 # Constant Rating = 4.0 (Baik)
                    try:
                        sim.compute()
                        z_grid[i, j] = sim.output['besar_diskon']
                    except:
                        z_grid[i, j] = 25.0
            return x_grid, y_grid, z_grid

        x_grid, y_grid, z_grid = build_interactive_surface(surf_key)
        
        fig_surf = go.Figure(data=[go.Surface(
            z=z_grid, x=x_grid, y=y_grid, 
            colorscale='Viridis',
            colorbar=dict(title='Diskon (%)', thickness=15)
        )])
        fig_surf.update_layout(
            title=f'Permukaan Kontrol 3D: Stok vs Penjualan (Rating = 4.0, Model: {surf_model})',
            scene=dict(
                xaxis_title='Stock Level (Rank)',
                yaxis_title='Jumlah Penjualan (Rank)',
                zaxis_title='Diskon (%)'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=600,
            width=900
        )
        st.plotly_chart(fig_surf, use_container_width=True)
        st.caption("💡 Petunjuk: Klik dan seret mouse pada gambar 3D di atas untuk memutar grafik, dan scroll untuk memperbesar/memperkecil.")

    with tab_pregenerated:
        st.markdown("### Gambar Visualisasi Hasil Pengujian")
        st.write("Di bawah ini adalah gambar analisis univariat, distribusi, dan matriks kebingungan yang disimpan di direktori output proyek.")
        
        output_images = {
            "univariate_stok_comparison.png": "Distribusi Stok Level: linear vs log vs rank-based",
            "comparison_tier_distribution.png": "Perbandingan Distribusi Tier Diskon: Sebelum vs Sesudah Kalibrasi",
            "evaluation_confusion_matrix.png": "Confusion Matrix Model Terkalibrasi (Ground Truth Ekspert)",
            "evaluation_sensitivity_boxplot.png": "Analisis Sensitivitas Fluktuasi Stok Level terhadap Output Diskon"
        }
        
        # Layout in 2x2 grid
        img_names = list(output_images.keys())
        c_i1, c_i2 = st.columns(2)
        
        for i, img_name in enumerate(img_names):
            full_path = os.path.join("output", img_name)
            desc = output_images[img_name]
            col_sel = c_i1 if i % 2 == 0 else c_i2
            
            with col_sel:
                st.markdown(f"##### 📸 {desc}")
                if os.path.exists(full_path):
                    st.image(full_path, use_container_width=True)
                else:
                    st.warning(f"File gambar `{full_path}` tidak ditemukan.")

# ==============================================================================
# PAGE 4: RULES & METHODOLOGY
# ==============================================================================
elif page == "📚 Aturan & Metodologi":
    st.markdown('<h1 class="main-title">Aturan Bisnis & Metodologi Fuzzy</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Rincian parameter linguistik, basis aturan keputusan, dan justifikasi ilmiah.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Landasan Teori Mamdani FIS
    Sistem rekomendasi diskon produk e-commerce ini mengadopsi algoritma **Mamdani Fuzzy Inference System**, yang mencakup empat fase utama:
    1. **Fuzzifikasi:** Mentransformasikan nilai numerik tegas (*crisp inputs*) menjadi derajat keanggotaan linguistik menggunakan fungsi keanggotaan segitiga (*trimf*).
    2. **Evaluasi Aturan:** Mengoperasikan operator logika (AND menggunakan fungsi minimum) pada basis aturan formal.
    3. **Agregasi:** Menggabungkan output dari aturan-aturan yang teraktivasi menjadi satu distribusi fuzzy akumulatif.
    4. **Defuzzifikasi:** Mengonversi hasil fuzzy agregat menjadi nilai numerik diskon tegas (0% - 50%) menggunakan metode **Centroid (Center of Area)**.
    """)
    
    st.markdown("### 📋 Tabel Basis Aturan Bisnis (8 Aturan)")
    st.write("Aturan-aturan di bawah ini disusun berdasarkan justifikasi akademis manajemen rantai pasok dan penetapan harga ritel ritel:")
    
    # 8 rules formatted table
    rules_table = [
        {"No": "1", "Stok": "Banyak", "Penjualan": "Rendah", "Rating": "-", "Output Diskon": "Besar", "Justifikasi / Jurnal Pendukung": "Pembersihan gudang (liquidating slow-moving stocks) [Petrovic, 1999]"},
        {"No": "2", "Stok": "Sedikit", "Penjualan": "Tinggi", "Rating": "-", "Output Diskon": "Kecil", "Justifikasi / Jurnal Pendukung": "Produk populer stok terbatas tidak butuh diskon tinggi (opportunity loss protection) [Zhao, 2016]"},
        {"No": "3", "Stok": "-", "Penjualan": "-", "Rating": "Buruk", "Output Diskon": "Besar", "Justifikasi / Jurnal Pendukung": "Likuidasi reputasi buruk agar modal kembali cepat [Grewal, 1998]"},
        {"No": "4", "Stok": "Sedang", "Penjualan": "Sedang", "Rating": "-", "Output Diskon": "Sedang", "Justifikasi / Jurnal Pendukung": "Strategi harga stabil untuk menjaga marjin ritel [Kumar, 2016]"},
        {"No": "5", "Stok": "Banyak", "Penjualan": "Sedang", "Rating": "-", "Output Diskon": "Sedang", "Justifikasi / Jurnal Pendukung": "Diskon moderat untuk meningkatkan laju stok sedang [Renna, 2015]"},
        {"No": "6", "Stok": "Sedang", "Penjualan": "Rendah", "Rating": "-", "Output Diskon": "Sedang", "Justifikasi / Jurnal Pendukung": "Stimulus harga agar produk tidak masuk tier lambat [Petrovic, 1999]"},
        {"No": "7", "Stok": "Sedikit", "Penjualan": "Sedang", "Rating": "-", "Output Diskon": "Kecil", "Justifikasi / Jurnal Pendukung": "Menjaga stok tersisa tetap terjual pada harga wajar [Shekarian, 2017]"},
        {"No": "8", "Stok": "-", "Penjualan": "Tinggi", "Rating": "Baik", "Output Diskon": "Kecil", "Justifikasi / Jurnal Pendukung": "Produk bintang dengan penjualan tinggi, keuntungan maksimal [Kumar, 2016]"}
    ]
    st.table(rules_table)
    
    st.markdown("""
    ### 📕 Batasan Rentang Output Diskon (0% - 50%)
    Secara akademis, diskon produk dibatasi maksimal **50%** berdasarkan dua pertimbangan:
    - **Quality-Signaling Theory:** Diskon di atas 50% akan menurunkan persepsi pembeli terhadap kualitas produk secara drastis (*brand dilution*) [Grewal et al., 1998].
    - **Margin Protection:** Dari perspektif pemodelan rantai pasok ritel ritel, elastisitas volume penjualan tidak lagi efisien untuk mengompensasi penurunan margin keuntungan per unit yang sangat tajam di atas ambang batas 50% [Kumar et al., 2016].
    """)

# Footer logo / branding
st.markdown("---")
st.caption("💻 Dikembangkan oleh Antigravity Google DeepMind Computational Intelligence Division untuk Universitas Komputasi Ritel.")
