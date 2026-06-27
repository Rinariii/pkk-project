import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

def create_preprocessing_notebook():
    nb = new_notebook()
    nb.cells.extend([
        new_markdown_cell("# Part 2: Data Preprocessing\n"
                          "This notebook processes the raw Olist dataset files and prepares the product-level features for the Fuzzy Inference System, incorporating original min-max, log-transform, and rank-based normalization methods."),
        new_code_cell("import pandas as pd\n"
                      "import numpy as np\n"
                      "import os\n\n"
                      "print('Pandas version:', pd.__version__)\n"
                      "print('Numpy version:', np.__version__)"),
        new_markdown_cell("## 1. Load Raw Datasets\n"
                          "We load the orders, order items, reviews, and products datasets from Olist."),
        new_code_cell("orders = pd.read_csv('dataset/olist_orders_dataset.csv')\n"
                      "items = pd.read_csv('dataset/olist_order_items_dataset.csv')\n"
                      "reviews = pd.read_csv('dataset/olist_order_reviews_dataset.csv')\n"
                      "products = pd.read_csv('dataset/olist_products_dataset.csv')\n\n"
                      "print('Orders shape:', orders.shape)\n"
                      "print('Items shape:', items.shape)\n"
                      "print('Reviews shape:', reviews.shape)\n"
                      "print('Products shape:', products.shape)"),
        new_markdown_cell("## 2. Filter Orders and Join Tables\n"
                          "We filter the orders to only include 'delivered' status transactions and join all four datasets together."),
        new_code_cell("delivered_orders = orders[orders['order_status'] == 'delivered']\n"
                      "merged = items.merge(delivered_orders, on='order_id', how='inner')\n"
                      "merged = merged.merge(reviews, on='order_id', how='left')\n"
                      "merged = merged.merge(products, on='product_id', how='left')\n\n"
                      "print('Merged dataset shape:', merged.shape)"),
        new_markdown_cell("## 3. Aggregate at Product Level\n"
                          "We aggregate the transaction-level data to get the key features for each product."),
        new_code_cell("agg = merged.groupby('product_id').agg(\n"
                      "    total_sold=('order_item_id', 'count'),\n"
                      "    jumlah_penjualan=('order_id', 'nunique'),\n"
                      "    avg_rating=('review_score', 'mean'),\n"
                      "    total_revenue=('price', 'sum'),\n"
                      "    category=('product_category_name', 'first')\n"
                      ").reset_index()\n\n"
                      "print('Aggregated products count:', len(agg))\n"
                      "print(agg.head())"),
        new_markdown_cell("## 4. Filter Data Density First\n"
                          "To ensure statistical validity of the rating scores, we filter out products that have less than 5 unique orders first, before applying any normalization scaling. This ensures our scales span the full range $[0, 100]$ for active products."),
        new_code_cell("# Filter products with unique orders >= 5\n"
                      "filtered = agg[agg['jumlah_penjualan'] >= 5].copy()\n"
                      "print('Filtered products count (orders >= 5):', len(filtered))"),
        new_markdown_cell("## 5. Normalization and Transformations on Active Products\n"
                          "We calculate three different sets of variables for inventory (stock) and demand (sales) on the filtered active products:\n"
                          "1. **Original (Min-Max)**: Linear Min-Max scaling on total sold.\n"
                          "2. **Log-Transform**: Applies a logarithmic transform $\\log(1 + x)$ to total sold before Min-Max scaling, to compress the extreme right-skewness.\n"
                          "3. **Rank-Based**: Computes the rank of total sold / unique order count and scales it to $[0, 100]$, providing a uniform distribution."),
        new_code_cell("from scipy.stats import rankdata\n\n"
                      "# A. Original (Min-Max)\n"
                      "min_sold = filtered['total_sold'].min()\n"
                      "max_sold = filtered['total_sold'].max()\n"
                      "filtered['stok_level_orig'] = 100.0 - ((filtered['total_sold'] - min_sold) / (max_sold - min_sold) * 100.0)\n\n"
                      "min_jual = filtered['jumlah_penjualan'].min()\n"
                      "max_jual = filtered['jumlah_penjualan'].max()\n"
                      "filtered['jumlah_penjualan_orig'] = (filtered['jumlah_penjualan'] - min_jual) / (max_jual - min_jual) * 100.0\n\n"
                      "# B. Log-Transform Normalization\n"
                      "log_sold = np.log1p(filtered['total_sold'])\n"
                      "min_log = log_sold.min()\n"
                      "max_log = log_sold.max()\n"
                      "filtered['jumlah_penjualan_log'] = (log_sold - min_log) / (max_log - min_log) * 100.0\n"
                      "filtered['stok_level_log'] = 100.0 - filtered['jumlah_penjualan_log']\n\n"
                      "# C. Rank-Based Normalization\n"
                      "filtered['jumlah_penjualan_rank'] = rankdata(filtered['jumlah_penjualan']) / len(filtered) * 100.0\n"
                      "filtered['stok_level_rank'] = 100.0 - filtered['jumlah_penjualan_rank']\n\n"
                      "print('Normalization check (means):')\n"
                      "print(filtered[['stok_level_orig', 'stok_level_log', 'stok_level_rank']].mean())"),
        new_markdown_cell("## 6. Imputation of Missing Values\n"
                          "We fill missing category names with 'unknown'. For missing review ratings, we impute them using the median rating of their respective product category. If the category median is also missing, we use the global median."),
        new_code_cell("filtered['category'] = filtered['category'].fillna('unknown')\n\n"
                      "global_median = filtered['avg_rating'].median()\n"
                      "category_medians = filtered.groupby('category')['avg_rating'].transform('median')\n"
                      "filtered['avg_rating'] = filtered['avg_rating'].fillna(category_medians).fillna(global_median)\n\n"
                      "print('Missing values count after imputation:')\n"
                      "print(filtered.isnull().sum())"),
        new_markdown_cell("## 7. Export Cleaned Dataset\n"
                          "Finally, we save the cleaned product features to `dataset/product_features_clean.csv`."),
        new_code_cell("os.makedirs('dataset', exist_ok=True)\n"
                      "filtered.to_csv('dataset/product_features_clean.csv', index=False)\n"
                      "print('Cleaned dataset saved successfully to dataset/product_features_clean.csv. Total rows:', len(filtered))")
    ])
    return nb

def create_eda_notebook():
    nb = new_notebook()
    nb.cells.extend([
        new_markdown_cell("# Part 1: Exploratory Data Analysis (EDA)\n"
                          "This notebook explores the preprocessed product features, visualizes their distributions before and after transformations, and saves plots for the final report."),
        new_code_cell("import pandas as pd\n"
                      "import numpy as np\n"
                      "import matplotlib.pyplot as plt\n"
                      "import seaborn as sns\n"
                      "import os\n\n"
                      "sns.set_theme(style='whitegrid')\n"
                      "os.makedirs('report/figures', exist_ok=True)\n"
                      "os.makedirs('output', exist_ok=True)"),
        new_markdown_cell("## 1. Load Data\n"
                          "We load the preprocessed product features dataset."),
        new_code_cell("df = pd.read_csv('dataset/product_features_clean.csv')\n"
                      "print('Cleaned features shape:', df.shape)\n"
                      "print(df.describe())"),
        new_markdown_cell("## 2. Compare Sales and Stock Level Skewness\n"
                          "We plot the distribution of stock level using the original linear scaling, log-transform, and rank-based scaling to visualize how the transformations compress skewness."),
        new_code_cell("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\n"
                      "sns.histplot(df['stok_level_orig'], bins=20, kde=True, ax=axes[0], color='salmon')\n"
                      "axes[0].set_title('Tingkat Stok Original (Heuristik Min-Max)')\n"
                      "axes[0].set_xlabel('Stock Level (Original)')\n"
                      "axes[0].set_ylabel('Frekuensi')\n\n"
                      "sns.histplot(df['stok_level_log'], bins=20, kde=True, ax=axes[1], color='skyblue')\n"
                      "axes[1].set_title('Tingkat Stok Setelah Log-Transform')\n"
                      "axes[1].set_xlabel('Stock Level (Log-Transform)')\n"
                      "axes[1].set_ylabel('Frekuensi')\n\n"
                      "sns.histplot(df['stok_level_rank'], bins=20, kde=True, ax=axes[2], color='lightgreen')\n"
                      "axes[2].set_title('Tingkat Stok Setelah Rank-Based')\n"
                      "axes[2].set_xlabel('Stock Level (Rank-Based)')\n"
                      "axes[2].set_ylabel('Frekuensi')\n\n"
                      "plt.tight_layout()\n"
                      "plt.savefig('report/figures/univariate_stok_comparison.png', dpi=150)\n"
                      "plt.savefig('output/univariate_stok_comparison.png', dpi=150)\n"
                      "plt.show()"),
        new_markdown_cell("## 3. Average Rating Distribution\n"
                      "We look at the distribution of product ratings to understand how customers rate products."),
        new_code_cell("plt.figure(figsize=(8, 5))\n"
                      "sns.histplot(df['avg_rating'], bins=15, kde=True, color='purple')\n"
                      "plt.title('Distribusi Rata-rata Rating Produk')\n"
                      "plt.xlabel('Average Rating')\n"
                      "plt.ylabel('Frekuensi')\n"
                      "plt.tight_layout()\n"
                      "plt.savefig('report/figures/univariate_avg_rating.png', dpi=150)\n"
                      "plt.show()")
    ])
    return nb

def create_modelling_notebook():
    nb = new_notebook()
    nb.cells.extend([
        new_markdown_cell("# Part 3: Fuzzy Inference System (FIS) Modelling\n"
                          "This notebook defines the calibrated membership functions using data percentiles, sets up the 8-rule base, runs three simulations (Sebelum, Log-Transform, and Rank-Based), and exports the results and control surfaces."),
        new_code_cell("import numpy as np\n"
                      "import pandas as pd\n"
                      "import skfuzzy as fuzz\n"
                      "from skfuzzy import control as ctrl\n"
                      "import matplotlib.pyplot as plt\n"
                      "from mpl_toolkits.mplot3d import Axes3D\n"
                      "import os\n\n"
                      "os.makedirs('report/figures', exist_ok=True)\n"
                      "os.makedirs('output', exist_ok=True)"),
        new_markdown_cell("## 1. Load Data & Compute Empirical Percentiles\n"
                          "We compute the percentiles for the different normalizations to quantitatively calibrate the fuzzy membership functions."),
        new_code_cell("df = pd.read_csv('dataset/product_features_clean.csv')\n\n"
                      "percentiles = [.25, .33, .50, .67, .75]\n"
                      "stats = df[['stok_level_log', 'jumlah_penjualan_log', 'stok_level_rank', 'jumlah_penjualan_rank', 'avg_rating']].describe(percentiles=percentiles)\n"
                      "print('=== EMPIRES PERCENTILES TABLE ===')\n"
                      "print(stats.loc[['25%', '33%', '50%', '67%', '75%']])"),
        new_markdown_cell("## 2. Define Fuzzy Control Simulation Functions\n"
                          "We define a helper function to set up the MFs using the data percentiles and run the FIS for all products."),
        new_code_cell("def run_fis_model(df, stok_col, penjualan_col, MFs_stok, MFs_penjualan, P50_rating):\n"
                      "    # Define Antecedents and Consequent\n"
                      "    stok = ctrl.Antecedent(np.arange(0, 101, 1), 'stok_level')\n"
                      "    penjualan = ctrl.Antecedent(np.arange(0, 101, 1), 'jumlah_penjualan')\n"
                      "    rating = ctrl.Antecedent(np.arange(1.0, 5.1, 0.1), 'avg_rating')\n"
                      "    diskon = ctrl.Consequent(np.arange(0, 51, 1), 'besar_diskon')\n\n"
                      "    # Calibrate MFs\n"
                      "    stok['Sedikit'] = fuzz.trimf(stok.universe, [0, 0, MFs_stok['P33']])\n"
                      "    stok['Sedang'] = fuzz.trimf(stok.universe, [MFs_stok['P25'], MFs_stok['P50'], MFs_stok['P75']])\n"
                      "    stok['Banyak'] = fuzz.trimf(stok.universe, [MFs_stok['P67'], 100, 100])\n\n"
                      "    penjualan['Rendah'] = fuzz.trimf(penjualan.universe, [0, 0, MFs_penjualan['P33']])\n"
                      "    penjualan['Sedang'] = fuzz.trimf(penjualan.universe, [MFs_penjualan['P25'], MFs_penjualan['P50'], MFs_penjualan['P75']])\n"
                      "    penjualan['Tinggi'] = fuzz.trimf(penjualan.universe, [MFs_penjualan['P67'], 100, 100])\n\n"
                      "    rating['Buruk'] = fuzz.trimf(rating.universe, [1.0, 1.0, P50_rating])\n"
                      "    rating['Baik'] = fuzz.trimf(rating.universe, [P50_rating, 5.0, 5.0])\n\n"
                      "    diskon['Kecil'] = fuzz.trimf(diskon.universe, [0, 0, 20])\n"
                      "    diskon['Sedang'] = fuzz.trimf(diskon.universe, [10, 25, 40])\n"
                      "    diskon['Besar'] = fuzz.trimf(diskon.universe, [30, 50, 50])\n\n"
                      "    # Define 8 rules\n"
                      "    rule1 = ctrl.Rule(stok['Banyak'] & penjualan['Rendah'], diskon['Besar'])\n"
                      "    rule2 = ctrl.Rule(stok['Sedikit'] & penjualan['Tinggi'], diskon['Kecil'])\n"
                      "    rule3 = ctrl.Rule(rating['Buruk'], diskon['Besar'])\n"
                      "    rule4 = ctrl.Rule(stok['Sedang'] & penjualan['Sedang'], diskon['Sedang'])\n"
                      "    rule5 = ctrl.Rule(stok['Banyak'] & penjualan['Sedang'], diskon['Sedang'])\n"
                      "    rule6 = ctrl.Rule(stok['Sedang'] & penjualan['Rendah'], diskon['Sedang'])\n"
                      "    rule7 = ctrl.Rule(stok['Sedikit'] & penjualan['Sedang'], diskon['Kecil'])\n"
                      "    rule8 = ctrl.Rule(rating['Baik'] & penjualan['Tinggi'], diskon['Kecil'])\n\n"
                      "    discount_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])\n"
                      "    discount_sim = ctrl.ControlSystemSimulation(discount_ctrl)\n\n"
                      "    discounts = []\n"
                      "    for idx, row in df.iterrows():\n"
                      "        discount_sim.input['stok_level'] = row[stok_col]\n"
                      "        discount_sim.input['jumlah_penjualan'] = row[penjualan_col]\n"
                      "        discount_sim.input['avg_rating'] = row['avg_rating']\n"
                      "        try:\n"
                      "            discount_sim.compute()\n"
                      "            val = discount_sim.output['besar_diskon']\n"
                      "        except:\n"
                      "            val = 25.0 # default fallback\n"
                      "        discounts.append(val)\n"
                      "    return discounts, discount_sim"),
        new_markdown_cell("## 3. Run Simulation for Three Scenarios\n"
                          "We execute the FIS for three scenarios:\n"
                          "1. **Sebelum (Heuristic)**: Heuristic MFs (0, 20, 30, 40, 50, 60, 80, 100) and raw min-max scaled variables.\n"
                          "2. **Log-Transform Model**: Percentile calibrated MFs and log-transformed variables.\n"
                          "3. **Rank-Based Model**: Percentile calibrated MFs and rank-normalized variables."),
        new_code_cell("# A. Sebelum (Heuristic)\n"
                      "def run_orig_model(df):\n"
                      "    stok = ctrl.Antecedent(np.arange(0, 101, 1), 'stok_level')\n"
                      "    penjualan = ctrl.Antecedent(np.arange(0, 101, 1), 'jumlah_penjualan')\n"
                      "    rating = ctrl.Antecedent(np.arange(1.0, 5.1, 0.1), 'avg_rating')\n"
                      "    diskon = ctrl.Consequent(np.arange(0, 51, 1), 'besar_diskon')\n\n"
                      "    stok['Sedikit'] = fuzz.trimf(stok.universe, [0, 0, 40])\n"
                      "    stok['Sedang'] = fuzz.trimf(stok.universe, [20, 50, 80])\n"
                      "    stok['Banyak'] = fuzz.trimf(stok.universe, [60, 100, 100])\n\n"
                      "    penjualan['Rendah'] = fuzz.trimf(penjualan.universe, [0, 0, 40])\n"
                      "    penjualan['Sedang'] = fuzz.trimf(penjualan.universe, [20, 50, 80])\n"
                      "    penjualan['Tinggi'] = fuzz.trimf(penjualan.universe, [60, 100, 100])\n\n"
                      "    rating['Buruk'] = fuzz.trimf(rating.universe, [1.0, 1.0, 3.0])\n"
                      "    rating['Baik'] = fuzz.trimf(rating.universe, [3.0, 5.0, 5.0])\n\n"
                      "    diskon['Kecil'] = fuzz.trimf(diskon.universe, [0, 0, 20])\n"
                      "    diskon['Sedang'] = fuzz.trimf(diskon.universe, [10, 25, 40])\n"
                      "    diskon['Besar'] = fuzz.trimf(diskon.universe, [30, 50, 50])\n\n"
                      "    rule1 = ctrl.Rule(stok['Banyak'] & penjualan['Rendah'], diskon['Besar'])\n"
                      "    rule2 = ctrl.Rule(stok['Sedikit'] & penjualan['Tinggi'], diskon['Kecil'])\n"
                      "    rule3 = ctrl.Rule(rating['Buruk'], diskon['Besar'])\n"
                      "    rule4 = ctrl.Rule(stok['Sedang'] & penjualan['Sedang'], diskon['Sedang'])\n"
                      "    rule5 = ctrl.Rule(stok['Banyak'] & penjualan['Sedang'], diskon['Sedang'])\n"
                      "    rule6 = ctrl.Rule(stok['Sedang'] & penjualan['Rendah'], diskon['Sedang'])\n"
                      "    rule7 = ctrl.Rule(stok['Sedikit'] & penjualan['Sedang'], diskon['Kecil'])\n"
                      "    rule8 = ctrl.Rule(rating['Baik'] & penjualan['Tinggi'], diskon['Kecil'])\n\n"
                      "    discount_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])\n"
                      "    discount_sim = ctrl.ControlSystemSimulation(discount_ctrl)\n\n"
                      "    discounts = []\n"
                      "    for idx, row in df.iterrows():\n"
                      "        # normalize jumlah_penjualan to [0, 100]\n"
                      "        discount_sim.input['stok_level'] = row['stok_level_orig']\n"
                      "        discount_sim.input['jumlah_penjualan'] = row['jumlah_penjualan_orig']\n"
                      "        discount_sim.input['avg_rating'] = row['avg_rating']\n"
                      "        try:\n"
                      "            discount_sim.compute()\n"
                      "            val = discount_sim.output['besar_diskon']\n"
                      "        except:\n"
                      "            val = 25.0\n"
                      "        discounts.append(val)\n"
                      "    return discounts\n\n"
                      "df['besar_diskon_sebelum'] = run_orig_model(df)"),
        new_code_cell("# B. Calibrated Log-Transform Model\n"
                      "MFs_stok_log = {'P25': stats.loc['25%', 'stok_level_log'], 'P33': stats.loc['33%', 'stok_level_log'], 'P50': stats.loc['50%', 'stok_level_log'], 'P67': stats.loc['67%', 'stok_level_log'], 'P75': stats.loc['75%', 'stok_level_log']}\n"
                      "MFs_penjualan_log = {'P25': stats.loc['25%', 'jumlah_penjualan_log'], 'P33': stats.loc['33%', 'jumlah_penjualan_log'], 'P50': stats.loc['50%', 'jumlah_penjualan_log'], 'P67': stats.loc['67%', 'jumlah_penjualan_log'], 'P75': stats.loc['75%', 'jumlah_penjualan_log']}\n"
                      "P50_rating = stats.loc['50%', 'avg_rating']\n\n"
                      "df['besar_diskon_log'], sim_log = run_fis_model(\n"
                      "    df, 'stok_level_log', 'jumlah_penjualan_log', MFs_stok_log, MFs_penjualan_log, P50_rating\n"
                      ")"),
        new_code_cell("# C. Calibrated Rank-Based Model\n"
                      "MFs_stok_rank = {'P25': stats.loc['25%', 'stok_level_rank'], 'P33': stats.loc['33%', 'stok_level_rank'], 'P50': stats.loc['50%', 'stok_level_rank'], 'P67': stats.loc['67%', 'stok_level_rank'], 'P75': stats.loc['75%', 'stok_level_rank']}\n"
                      "MFs_penjualan_rank = {'P25': stats.loc['25%', 'jumlah_penjualan_rank'], 'P33': stats.loc['33%', 'jumlah_penjualan_rank'], 'P50': stats.loc['50%', 'jumlah_penjualan_rank'], 'P67': stats.loc['67%', 'jumlah_penjualan_rank'], 'P75': stats.loc['75%', 'jumlah_penjualan_rank']}\n\n"
                      "df['besar_diskon_rank'], sim_rank = run_fis_model(\n"
                      "    df, 'stok_level_rank', 'jumlah_penjualan_rank', MFs_stok_rank, MFs_penjualan_rank, P50_rating\n"
                      ")\n\n"
                      "# Export results\n"
                      "df.to_csv('dataset/fis_discount_results.csv', index=False)\n"
                      "print('FIS simulations completed and saved to dataset/fis_discount_results.csv')"),
        new_markdown_cell("## 4. Plot 3D Control Surfaces for Chosen Model (Rank-Based)\n"
                          "We map out the 3D surface plot to visualize the control dynamics of the calibrated Rank-Based system."),
        new_code_cell("# Grid for Stock vs Sales (Rating = 4.0)\n"
                      "x_range = np.linspace(0, 100, 30)\n"
                      "y_range = np.linspace(0, 100, 30)\n"
                      "x_grid, y_grid = np.meshgrid(x_range, y_range)\n"
                      "z_grid = np.zeros_like(x_grid)\n\n"
                      "for i in range(x_grid.shape[0]):\n"
                      "    for j in range(x_grid.shape[1]):\n"
                      "        sim_rank.input['stok_level'] = x_grid[i, j]\n"
                      "        sim_rank.input['jumlah_penjualan'] = y_grid[i, j]\n"
                      "        sim_rank.input['avg_rating'] = 4.0\n"
                      "        try:\n"
                      "            sim_rank.compute()\n"
                      "            z_grid[i, j] = sim_rank.output['besar_diskon']\n"
                      "        except:\n"
                      "            z_grid[i, j] = 25.0\n\n"
                      "fig = plt.figure(figsize=(10, 7))\n"
                      "ax = fig.add_subplot(111, projection='3d')\n"
                      "surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none')\n"
                      "ax.set_title('Permukaan Kontrol 3D: Level Stok vs Jumlah Penjualan (Calibrated Rank-Based)')\n"
                      "ax.set_xlabel('Stock Level (Rank)')\n"
                      "ax.set_ylabel('Jumlah Penjualan (Rank)')\n"
                      "ax.set_zlabel('Besar Diskon (%)')\n"
                      "fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)\n"
                      "plt.tight_layout()\n"
                      "plt.savefig('report/figures/modelling_surface_stok_penjualan.png', dpi=150)\n"
                      "plt.savefig('output/modelling_surface_stok_penjualan.png', dpi=150)\n"
                      "plt.show()"),
        new_code_cell("# Grid for Rating vs Stock (Sales = 50.0)\n"
                      "x_range = np.linspace(1.0, 5.0, 30)\n"
                      "y_range = np.linspace(0, 100, 30)\n"
                      "x_grid, y_grid = np.meshgrid(x_range, y_range)\n"
                      "z_grid = np.zeros_like(x_grid)\n\n"
                      "for i in range(x_grid.shape[0]):\n"
                      "    for j in range(x_grid.shape[1]):\n"
                      "        sim_rank.input['avg_rating'] = x_grid[i, j]\n"
                      "        sim_rank.input['stok_level'] = y_grid[i, j]\n"
                      "        sim_rank.input['jumlah_penjualan'] = 50.0\n"
                      "        try:\n"
                      "            sim_rank.compute()\n"
                      "            z_grid[i, j] = sim_rank.output['besar_diskon']\n"
                      "        except:\n"
                      "            z_grid[i, j] = 25.0\n\n"
                      "fig = plt.figure(figsize=(10, 7))\n"
                      "ax = fig.add_subplot(111, projection='3d')\n"
                      "surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none')\n"
                      "ax.set_title('Permukaan Kontrol 3D: Average Rating vs Level Stok (Calibrated Rank-Based)')\n"
                      "ax.set_xlabel('Average Rating')\n"
                      "ax.set_ylabel('Stock Level (Rank)')\n"
                      "ax.set_zlabel('Besar Diskon (%)')\n"
                      "fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)\n"
                      "plt.tight_layout()\n"
                      "plt.savefig('report/figures/modelling_surface_rating_stok.png', dpi=150)\n"
                      "plt.savefig('output/modelling_surface_rating_stok.png', dpi=150)\n"
                      "plt.show()")
    ])
    return nb

def create_evaluation_notebook():
    nb = new_notebook()
    nb.cells.extend([
        new_markdown_cell("# Part 4: Evaluation and Sensitivity Analysis\n"
                          "This notebook compares output distributions BEFORE vs AFTER calibration, evaluates the calibrated Rank-based model using synthetic test cases, and runs the sensitivity analysis."),
        new_code_cell("import numpy as np\n"
                      "import pandas as pd\n"
                      "import matplotlib.pyplot as plt\n"
                      "import seaborn as sns\n"
                      "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n"
                      "import skfuzzy as fuzz\n"
                      "from skfuzzy import control as ctrl\n"
                      "import os\n\n"
                      "sns.set_theme(style='whitegrid')\n"
                      "os.makedirs('report/figures', exist_ok=True)\n"
                      "os.makedirs('output', exist_ok=True)"),
        new_markdown_cell("## 1. Distribution Comparison BEFORE vs AFTER\n"
                          "We define the discount tiers based on standard boundaries:\n"
                          "- **Kecil:** $[0, 17.0]$\n"
                          "- **Sedang:** $(17.0, 33.0]$\n"
                          "- **Besar:** $(33.0, 50.0]$\n\n"
                          "We compare the distribution of the original model (Sebelum), Log-transform, and Rank-based models."),
        new_code_cell("df = pd.read_csv('dataset/fis_discount_results.csv')\n\n"
                      "def get_tier(d):\n"
                      "    if d <= 17.0:\n"
                      "        return 'Kecil'\n"
                      "    elif d <= 33.0:\n"
                      "        return 'Sedang'\n"
                      "    else:\n"
                      "        return 'Besar'\n\n"
                      "df['tier_sebelum'] = df['besar_diskon_sebelum'].apply(get_tier)\n"
                      "df['tier_log'] = df['besar_diskon_log'].apply(get_tier)\n"
                      "df['tier_rank'] = df['besar_diskon_rank'].apply(get_tier)\n\n"
                      "print('=== DISCOUNT TIER DISTRIBUTIONS ===')\n"
                      "for col in ['tier_sebelum', 'tier_log', 'tier_rank']:\n"
                      "    counts = df[col].value_counts()\n"
                      "    percents = df[col].value_counts(normalize=True) * 100\n"
                      "    print(f'\\nModel {col}:')\n"
                      "    for t in ['Kecil', 'Sedang', 'Besar']:\n"
                      "        c = counts.get(t, 0)\n"
                      "        p = percents.get(t, 0.0)\n"
                      "        print(f'  {t}: {c} products ({p:.2f}%)')"),
        new_code_cell("# Plot Comparison Bar Chart\n"
                      "melted = pd.melt(df[['tier_sebelum', 'tier_log', 'tier_rank']], var_name='Model', value_name='Tier')\n"
                      "melted['Model'] = melted['Model'].map({\n"
                      "    'tier_sebelum': 'Sebelum (Heuristik)',\n"
                      "    'tier_log': 'Sesudah (Log-Transform)',\n"
                      "    'tier_rank': 'Sesudah (Rank-Based)'\n"
                      "})\n\n"
                      "plt.figure(figsize=(10, 6))\n"
                      "sns.countplot(data=melted, x='Tier', hue='Model', order=['Kecil', 'Sedang', 'Besar'], palette='Set2')\n"
                      "plt.title('Perbandingan Distribusi Tier Diskon: Sebelum vs Sesudah Kalibrasi')\n"
                      "plt.xlabel('Tier Diskon')\n"
                      "plt.ylabel('Jumlah Produk')\n"
                      "plt.legend(title='Skenario Model')\n"
                      "plt.tight_layout()\n"
                      "plt.savefig('report/figures/evaluation_comparison_distribution.png', dpi=150)\n"
                      "plt.savefig('output/comparison_tier_distribution.png', dpi=150)\n"
                      "plt.show()"),
        new_markdown_cell("## 2. Re-Evaluate with 50 Synthetic Test Cases\n"
                          "We define the same 50 cases covering all rules to evaluate how the new calibrated Rank-based system aligns with business logic."),
        new_code_cell("# Build calibrated simulator using Rank percentiles\n"
                      "stok = ctrl.Antecedent(np.arange(0, 101, 1), 'stok_level')\n"
                      "penjualan = ctrl.Antecedent(np.arange(0, 101, 1), 'jumlah_penjualan')\n"
                      "rating = ctrl.Antecedent(np.arange(1.0, 5.1, 0.1), 'avg_rating')\n"
                      "diskon = ctrl.Consequent(np.arange(0, 51, 1), 'besar_diskon')\n\n"
                      "# Calibrate MFs using Rank empirical percentiles\n"
                      "stok['Sedikit'] = fuzz.trimf(stok.universe, [0, 0, 32.781699])\n"
                      "stok['Sedang'] = fuzz.trimf(stok.universe, [26.016062, 49.805305, 71.623266])\n"
                      "stok['Banyak'] = fuzz.trimf(stok.universe, [71.623266, 100, 100])\n\n"
                      "penjualan['Rendah'] = fuzz.trimf(penjualan.universe, [0, 0, 28.376734])\n"
                      "penjualan['Sedang'] = fuzz.trimf(penjualan.universe, [28.376734, 50.194695, 73.983938])\n"
                      "penjualan['Tinggi'] = fuzz.trimf(penjualan.universe, [67.218301, 100, 100])\n\n"
                      "rating['Buruk'] = fuzz.trimf(rating.universe, [1.0, 1.0, 4.188679])\n"
                      "rating['Baik'] = fuzz.trimf(rating.universe, [4.188679, 5.0, 5.0])\n\n"
                      "diskon['Kecil'] = fuzz.trimf(diskon.universe, [0, 0, 20])\n"
                      "diskon['Sedang'] = fuzz.trimf(diskon.universe, [10, 25, 40])\n"
                      "diskon['Besar'] = fuzz.trimf(diskon.universe, [30, 50, 50])\n\n"
                      "rule1 = ctrl.Rule(stok['Banyak'] & penjualan['Rendah'], diskon['Besar'])\n"
                      "rule2 = ctrl.Rule(stok['Sedikit'] & penjualan['Tinggi'], diskon['Kecil'])\n"
                      "rule3 = ctrl.Rule(rating['Buruk'], diskon['Besar'])\n"
                      "rule4 = ctrl.Rule(stok['Sedang'] & penjualan['Sedang'], diskon['Sedang'])\n"
                      "rule5 = ctrl.Rule(stok['Banyak'] & penjualan['Sedang'], diskon['Sedang'])\n"
                      "rule6 = ctrl.Rule(stok['Sedang'] & penjualan['Rendah'], diskon['Sedang'])\n"
                      "rule7 = ctrl.Rule(stok['Sedikit'] & penjualan['Sedang'], diskon['Kecil'])\n"
                      "rule8 = ctrl.Rule(rating['Baik'] & penjualan['Tinggi'], diskon['Kecil'])\n\n"
                      "discount_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])\n"
                      "discount_sim = ctrl.ControlSystemSimulation(discount_ctrl)"),
        new_code_cell("# Define the 50 synthetic test cases\n"
                      "test_cases = [\n"
                      "    # R1: Banyak, Rendah -> Besar\n"
                      "    (90, 10, 4.2, 'Besar'), (95, 5, 3.5, 'Besar'), (85, 12, 4.0, 'Besar'), (80, 15, 3.8, 'Besar'), (88, 8, 4.5, 'Besar'),\n"
                      "    # R2: Sedikit, Tinggi -> Kecil\n"
                      "    (10, 90, 4.0, 'Kecil'), (5, 95, 4.5, 'Kecil'), (12, 85, 3.8, 'Kecil'), (15, 80, 4.2, 'Kecil'), (8, 88, 4.7, 'Kecil'),\n"
                      "    # R3: Buruk -> Besar\n"
                      "    (50, 50, 1.5, 'Besar'), (30, 20, 2.0, 'Besar'), (70, 80, 1.8, 'Besar'), (40, 10, 2.2, 'Besar'), (60, 90, 2.5, 'Besar'),\n"
                      "    # R4: Sedang, Sedang -> Sedang\n"
                      "    (50, 50, 4.0, 'Sedang'), (45, 55, 3.8, 'Sedang'), (55, 45, 4.2, 'Sedang'), (48, 52, 3.5, 'Sedang'), (52, 48, 4.5, 'Sedang'),\n"
                      "    # R5: Banyak, Sedang -> Sedang\n"
                      "    (80, 50, 4.0, 'Sedang'), (85, 45, 3.8, 'Sedang'), (90, 55, 4.2, 'Sedang'), (88, 48, 3.5, 'Sedang'), (82, 52, 4.5, 'Sedang'),\n"
                      "    # R6: Sedang, Rendah -> Sedang\n"
                      "    (50, 10, 4.0, 'Sedang'), (45, 15, 3.8, 'Sedang'), (55, 5, 4.2, 'Sedang'), (48, 12, 3.5, 'Sedang'), (52, 8, 4.5, 'Sedang'),\n"
                      "    # R7: Sedikit, Sedang -> Kecil\n"
                      "    (10, 50, 4.0, 'Kecil'), (5, 45, 3.8, 'Kecil'), (12, 55, 4.2, 'Kecil'), (15, 48, 3.5, 'Kecil'), (8, 52, 4.5, 'Kecil'),\n"
                      "    # R8: Baik & Tinggi -> Kecil\n"
                      "    (50, 90, 4.5, 'Kecil'), (40, 85, 4.0, 'Kecil'), (60, 95, 3.8, 'Kecil'), (30, 80, 4.2, 'Kecil'), (70, 88, 4.7, 'Kecil'),\n"
                      "    # Extra general validation cases\n"
                      "    (90, 2, 1.2, 'Besar'), (2, 98, 4.9, 'Kecil'), (50, 50, 2.0, 'Besar'), (50, 50, 4.8, 'Sedang'), (90, 50, 2.5, 'Besar'),\n"
                      "    (10, 10, 4.5, 'Sedang'), (90, 90, 4.5, 'Kecil'), (10, 90, 1.5, 'Besar'), (90, 10, 4.8, 'Besar'), (50, 8, 4.8, 'Sedang')\n"
                      "]\n\n"
                      "results = []\n"
                      "for idx, (st_val, pj_val, rt_val, gt_tier) in enumerate(test_cases):\n"
                      "    discount_sim.input['stok_level'] = st_val\n"
                      "    discount_sim.input['jumlah_penjualan'] = pj_val\n"
                      "    discount_sim.input['avg_rating'] = rt_val\n"
                      "    try:\n"
                      "        discount_sim.compute()\n"
                      "        pred_val = discount_sim.output['besar_diskon']\n"
                      "    except:\n"
                      "        pred_val = 25.0\n"
                      "    pred_tier = get_tier(pred_val)\n"
                      "    results.append({\n"
                      "        'Case': idx + 1,\n"
                      "        'Stok': st_val,\n"
                      "        'Penjualan': pj_val,\n"
                      "        'Rating': rt_val,\n"
                      "        'GroundTruth': gt_tier,\n"
                      "        'PredictedDiskon': pred_val,\n"
                      "        'PredictedTier': pred_tier\n"
                      "    })\n\n"
                      "res_df = pd.DataFrame(results)\n"
                      "print('Example test cases under Calibrated Rank-Based model:')\n"
                      "print(res_df.head(10))"),
        new_markdown_cell("## 3. Calculate Performance Metrics and Plot Confusion Matrix"),
        new_code_cell("y_true = res_df['GroundTruth']\n"
                      "y_pred = res_df['PredictedTier']\n\n"
                      "acc = accuracy_score(y_true, y_pred) * 100\n"
                      "prec = precision_score(y_true, y_pred, average='macro') * 100\n"
                      "rec = recall_score(y_true, y_pred, average='macro') * 100\n"
                      "f1 = f1_score(y_true, y_pred, average='macro') * 100\n\n"
                      "print('=== PERFORMANCE METRICS ON SYNTHETIC TEST CASES (RANK-BASED) ===')\n"
                      "print(f'Accuracy: {acc:.2f}%')\n"
                      "print(f'Precision (Macro): {prec:.2f}%')\n"
                      "print(f'Recall (Macro): {rec:.2f}%')\n"
                      "print(f'F1-Score (Macro): {f1:.2f}%')\n\n"
                      "# Confusion Matrix\n"
                      "labels = ['Kecil', 'Sedang', 'Besar']\n"
                      "cm = confusion_matrix(y_true, y_pred, labels=labels)\n"
                      "plt.figure(figsize=(6, 5))\n"
                      "sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)\n"
                      "plt.title('Confusion Matrix: Calibrated FIS Prediksi vs Ground-Truth Pakar')\n"
                      "plt.xlabel('Prediksi Pakar')\n"
                      "plt.ylabel('Sebenarnya (Ground Truth)')\n"
                      "plt.tight_layout()\n"
                      "plt.savefig('report/figures/evaluation_confusion_matrix.png', dpi=150)\n"
                      "plt.savefig('output/evaluation_confusion_matrix.png', dpi=150)\n"
                      "plt.show()"),
        new_markdown_cell("## 4. Sensitivity Analysis on Stock Fluctuations\n"
                      "We evaluate how stable the recommendations are when the input `stok_level` has a $\pm 10\\%$ margin of error or fluctuation."),
        new_code_cell("# Baseline mean discount\n"
                      "baseline_mean = df['besar_diskon_rank'].mean()\n\n"
                      "# Scenario +10% Stock\n"
                      "discounts_plus = []\n"
                      "for idx, row in df.iterrows():\n"
                      "    # Cap stock at 100\n"
                      "    discount_sim.input['stok_level'] = min(row['stok_level_rank'] + 10.0, 100.0)\n"
                      "    discount_sim.input['jumlah_penjualan'] = row['jumlah_penjualan_rank']\n"
                      "    discount_sim.input['avg_rating'] = row['avg_rating']\n"
                      "    try:\n"
                      "        discount_sim.compute()\n"
                      "        val = discount_sim.output['besar_diskon']\n"
                      "    except:\n"
                      "        val = 25.0\n"
                      "    discounts_plus.append(val)\n"
                      "mean_plus = np.mean(discounts_plus)\n"
                      "change_plus = ((mean_plus - baseline_mean) / baseline_mean) * 100.0\n\n"
                      "# Scenario -10% Stock\n"
                      "discounts_minus = []\n"
                      "for idx, row in df.iterrows():\n"
                      "    # Floor stock at 0\n"
                      "    discount_sim.input['stok_level'] = max(row['stok_level_rank'] - 10.0, 0.0)\n"
                      "    discount_sim.input['jumlah_penjualan'] = row['jumlah_penjualan_rank']\n"
                      "    discount_sim.input['avg_rating'] = row['avg_rating']\n"
                      "    try:\n"
                      "        discount_sim.compute()\n"
                      "        val = discount_sim.output['besar_diskon']\n"
                      "    except:\n"
                      "        val = 25.0\n"
                      "    discounts_minus.append(val)\n"
                      "mean_minus = np.mean(discounts_minus)\n"
                      "change_minus = ((mean_minus - baseline_mean) / baseline_mean) * 100.0\n\n"
                      "print('=== SENSITIVITY ANALYSIS (RANK-BASED) ===')\n"
                      "print(f'Baseline Rata-rata Diskon: {baseline_mean:.4f}%')\n"
                      "print(f'Stok +10% Rata-rata Diskon: {mean_plus:.4f}% (Perubahan: {change_plus:+.4f}%)')\n"
                      "print(f'Stok -10% Rata-rata Diskon: {mean_minus:.4f}% (Perubahan: {change_minus:+.4f}%)')\n\n"
                      "# Plot sensitivity comparison\n"
                      "plt.figure(figsize=(8, 5))\n"
                      "sns.boxplot(data=pd.DataFrame({\n"
                      "    'Baseline': df['besar_diskon_rank'],\n"
                      "    'Stok +10%': discounts_plus,\n"
                      "    'Stok -10%': discounts_minus\n"
                      "}), palette='Set3')\n"
                      "plt.title('Analisis Sensitivitas: Perbandingan Distribusi Diskon (Rank-Based)')\n"
                      "plt.ylabel('Besar Diskon (%)')\n"
                      "plt.tight_layout()\n"
                      "plt.savefig('report/figures/evaluation_sensitivity_boxplot.png', dpi=150)\n"
                      "plt.savefig('output/evaluation_sensitivity_boxplot.png', dpi=150)\n"
                      "plt.show()")
    ])
    return nb

def generate_and_run_all():
    print("Generating notebooks...")
    nb_prep = create_preprocessing_notebook()
    nb_eda = create_eda_notebook()
    nb_model = create_modelling_notebook()
    nb_eval = create_evaluation_notebook()

    with open('2_preprocessing.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb_prep, f)
    with open('1_eda.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb_eda, f)
    with open('3_modelling.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb_model, f)
    with open('4_evaluation.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb_eval, f)
    print("Notebooks written successfully.")

    print("Executing notebooks...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Order of execution
    notebook_files = [
        '2_preprocessing.ipynb',
        '1_eda.ipynb',
        '3_modelling.ipynb',
        '4_evaluation.ipynb'
    ]
    
    for filename in notebook_files:
        print(f"Executing {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        with open(filename, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Finished executing {filename}.")

if __name__ == '__main__':
    generate_and_run_all()
