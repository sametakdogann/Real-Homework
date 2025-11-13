import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="B2B Intelligent Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 B2B Intelligent Sales Dashboard")

st.write(
    """
    This dashboard is built with **Streamlit** on top of a B2B transaction dataset.
    
    It provides:
    - Overall KPIs and revenue/quantity breakdowns  
    - Descriptive statistics and distributions  
    - An **ABC–XYZ stock classification** explorer  
    - A convenient filtered raw data browser  
    """
)

# -----------------------------
# SMALL HELPER FUNCTIONS
# -----------------------------
def format_currency(x: float) -> str:
    return f"{x:,.2f} ₺"

@st.cache_data
def load_data(uploaded_file):
    """
    Read Excel from uploader (or local file as fallback).

    If no file is uploaded (e.g. on teacher's computer),
    it tries to load `B2B_Transaction_Data.xlsx` from the same folder.
    """
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_excel("B2B_Transaction_Data.xlsx")
    return df

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔁 Data & Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload **B2B_Transaction_Data.xlsx**",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.sidebar.info(
        "You can upload the homework dataset here.\n\n"
        "If you are running this locally and the file is in the same folder "
        "as this script, the app will try to load it automatically."
    )

with st.sidebar.expander("ℹ About this dashboard", expanded=False):
    st.write(
        """
        - Built for **B2B sales analysis**  
        - Includes basic KPIs, descriptive stats,  
          and **ABC–XYZ inventory classification**.  
        - Use filters below to explore different segments.
        """
    )

# Try loading data (if file is really missing everywhere, fail gracefully)
try:
    df = load_data(uploaded_file)
except Exception:
    st.error("❌ Data could not be loaded. Please upload the Excel file or place it next to this script.")
    st.stop()

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# SalesRevenue = Quantity * UnitPrice
df["SalesRevenue"] = df["Quantity"] * df["UnitPrice"]

# Year, Month features
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.subheader("📅 Date Range")

min_date = df["InvoiceDate"].min()
max_date = df["InvoiceDate"].max()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = max_date

st.sidebar.subheader("🔎 Other Filters")

categories_all = sorted(df["Category"].dropna().unique())
cities_all = sorted(df["City"].dropna().unique())

categories = st.sidebar.multiselect(
    "Category",
    options=categories_all,
    default=categories_all
)

cities = st.sidebar.multiselect(
    "City",
    options=cities_all,
    default=cities_all
)

# Apply filters
df_filtered = df[
    (df["InvoiceDate"] >= pd.to_datetime(start_date)) &
    (df["InvoiceDate"] <= pd.to_datetime(end_date)) &
    (df["Category"].isin(categories)) &
    (df["City"].isin(cities))
].copy()

st.markdown("### ℹ Current filter summary")
st.write(
    f"**Date range:** *{start_date}* → *{end_date}*  |  "
    f"**Categories:** *{len(categories)} selected*  |  "
    f"**Cities:** *{len(cities)} selected*  |  "
    f"**Rows after filtering:** *{len(df_filtered):,}*"
)

if df_filtered.empty:
    st.warning("No data left after filtering. Please relax your filters from the sidebar.")
    st.stop()

# Quick download of filtered data
st.download_button(
    "⬇ Download filtered data as CSV",
    data=df_filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_b2b_transactions.csv",
    mime="text/csv",
    help="Exports the current filtered transaction-level dataset."
)

st.divider()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & KPIs",
    "Descriptive Statistics",
    "ABC–XYZ Analysis",
    "Raw Data"
])

# -----------------------------
# TAB 1: OVERVIEW & KPIs
# -----------------------------
with tab1:
    st.subheader("📌 Key Performance Indicators (KPIs)")

    total_revenue = df_filtered["SalesRevenue"].sum()
    total_quantity = df_filtered["Quantity"].sum()
    total_invoices = df_filtered["InvoiceNo"].nunique()
    total_customers = df_filtered["CustomerID"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", format_currency(total_revenue))
    col2.metric("Total Quantity Sold", f"{total_quantity:,.0f}")
    col3.metric("Number of Invoices", f"{total_invoices:,}")
    col4.metric("Number of Customers", f"{total_customers:,}")

    st.caption("KPIs are computed on **filtered** data only.")

    st.divider()

    # User can choose which metric to use in category & city charts
    st.markdown("#### Revenue / Quantity Breakdown")
    metric_for_breakdown = st.radio(
        "Choose metric for the next two charts:",
        options=["SalesRevenue", "Quantity"],
        index=0,
        horizontal=True,
        help="Switch between total revenue and total quantity."
    )

    metric_label = "Revenue" if metric_for_breakdown == "SalesRevenue" else "Quantity"

    # Layout for charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"##### {metric_label} by Category")
        revenue_by_cat = (
            df_filtered.groupby("Category")[metric_for_breakdown]
            .sum()
            .reset_index()
            .sort_values(metric_for_breakdown, ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(revenue_by_cat["Category"], revenue_by_cat[metric_for_breakdown])
        ax.set_title(f"Total {metric_label} by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel(metric_label)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown(f"##### {metric_label} by City (Top 10)")
        revenue_by_city = (
            df_filtered.groupby("City")[metric_for_breakdown]
            .sum()
            .reset_index()
            .sort_values(metric_for_breakdown, ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(revenue_by_city["City"], revenue_by_city[metric_for_breakdown])
        ax.set_title(f"Top 10 Cities by {metric_label}")
        ax.set_xlabel("City")
        ax.set_ylabel(metric_label)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("#### Monthly Revenue Trend")
    revenue_over_time = (
        df_filtered.groupby("Month")["SalesRevenue"]
        .sum()
        .reset_index()
        .sort_values("Month")
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(revenue_over_time["Month"], revenue_over_time["SalesRevenue"], marker="o")
    ax.set_title("Monthly Revenue Trend (Filtered Data)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Top 15 Products by Revenue")
    top_products = (
        df_filtered.groupby(["StockCode", "Description"])["SalesRevenue"]
        .sum()
        .reset_index()
        .sort_values("SalesRevenue", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top_products["Description"], top_products["SalesRevenue"])
    ax.set_title("Top 15 Products by Revenue")
    ax.set_xlabel("Product")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis='x', rotation=60)
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# TAB 2: DESCRIPTIVE STATISTICS
# -----------------------------
with tab2:
    st.subheader("📈 Descriptive Statistics")

    st.markdown("#### Numeric Columns Summary")
    numeric_cols = ["Quantity", "NetPrice", "UnitPrice", "SalesRevenue"]
    st.write(df_filtered[numeric_cols].describe().T)

    st.markdown("#### Distribution of a Numeric Variable")
    numeric_col = st.selectbox(
        "Select numeric column for distribution plots:",
        options=numeric_cols,
        index=numeric_cols.index("SalesRevenue") if "SalesRevenue" in numeric_cols else 0
    )

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_filtered[numeric_col].dropna(), bins=40)
    ax.set_title(f"Distribution of {numeric_col}")
    ax.set_xlabel(numeric_col)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Box Plot for Outlier Detection")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.boxplot(df_filtered[numeric_col].dropna(), vert=True)
    ax.set_title(f"Box Plot of {numeric_col}")
    ax.set_ylabel(numeric_col)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Correlation Heatmap (Numeric Features)")
    corr = df_filtered[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="Blues")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Matrix of Numeric Variables")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=8
            )

    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# TAB 3: ABC–XYZ ANALYSIS
# -----------------------------
with tab3:
    st.subheader("📦 ABC–XYZ Stock Classification")

    st.write(
        """
        ABC–XYZ analysis groups SKUs (**StockCode**) along two dimensions:
        - **ABC**: contribution to total revenue  
        - **XYZ**: demand variability (Coefficient of Variation of monthly sales)
        
        This supports inventory and supply chain decisions.
        """
    )

    # Optional: user-tunable thresholds
    st.markdown("##### Thresholds (you can tune if you want)")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        a_limit = col_t1.slider("A-class cutoff (cumulative %)", 0.5, 0.95, 0.80, 0.01)
    with col_t2:
        b_limit = col_t2.slider("B-class cutoff (cumulative %)", a_limit, 0.99, 0.95, 0.01)
    with col_t3:
        x_limit = col_t3.slider("X/Y CV boundary", 0.1, 1.0, 0.5, 0.05)
    y_limit = 1.0  # Y/Z boundary

    # 1) Month number from InvoiceDate
    df_abc = df_filtered.copy()
    df_abc["month_num"] = df_abc["InvoiceDate"].dt.month

    # 2) Monthly sales revenue per StockCode
    df_2 = (
        df_abc
        .groupby(["StockCode", "month_num"])["SalesRevenue"]
        .sum()
        .to_frame()
        .reset_index()
    )

    # 3) Pivot so that each month is a column
    df_3 = (
        df_2
        .pivot(index="StockCode", columns="month_num", values="SalesRevenue")
        .reset_index()
        .fillna(0)
    )

    # 4) Total sales, average monthly sales, standard deviation
    if df_3.shape[1] > 1:
        month_columns = df_3.columns[1:]  # all month columns
        df_3["total_sales"] = df_3[month_columns].sum(axis=1)
        df_3["average_sales"] = df_3["total_sales"] / len(month_columns)
        df_3["std_dev"] = df_3[month_columns].std(axis=1)

        # 5) Coefficient of variation (CV) = std / mean
        df_3["CV"] = np.where(
            df_3["average_sales"] > 0,
            df_3["std_dev"] / df_3["average_sales"],
            0.0
        )

        # 6) XYZ classification based on CV
        def xyz_analysis(x):
            if x <= x_limit:
                return "X"
            elif x_limit < x <= y_limit:
                return "Y"
            else:
                return "Z"

        df_3["XYZ_Class"] = df_3["CV"].apply(xyz_analysis)

        # 7) ABC based on total revenue (sum of total_sales)
        df_4 = (
            df_3.groupby("StockCode")
            .agg(total_revenue=("total_sales", "sum"))
            .sort_values(by="total_revenue", ascending=False)
            .reset_index()
        )

        # Cumulative percentages
        df_4["cumulative"] = df_4["total_revenue"].cumsum()
        df_4["total_cumulative"] = df_4["total_revenue"].sum()
        df_4["sku_percent"] = df_4["cumulative"] / df_4["total_cumulative"]

        # ABC classification function
        def abc_classification(p):
            if 0 < p <= a_limit:
                return "A"
            elif a_limit < p <= b_limit:
                return "B"
            else:
                return "C"

        df_4["ABC_Class"] = df_4["sku_percent"].apply(abc_classification)

        # 8) Merge ABC & XYZ info
        df_3_small = df_3[
            ["StockCode", "total_sales", "average_sales", "std_dev", "CV", "XYZ_Class"]
        ]
        df_4_small = df_4[
            ["StockCode", "total_revenue", "sku_percent", "ABC_Class"]
        ]

        df_final = df_4_small.merge(df_3_small, on="StockCode", how="left")

        # 9) Bring Description from original df
        df_desc = df_filtered[["StockCode", "Description"]].drop_duplicates()
        df_merge = df_final.merge(df_desc, on="StockCode", how="left")

        # 10) Remove duplicates and create final stock class
        df_result = df_merge.drop_duplicates().copy()
        df_result["stock_class"] = df_result["ABC_Class"].astype(str) + df_result["XYZ_Class"].astype(str)

        st.markdown("#### ABC–XYZ Summary Table")
        st.write(
            "Each row represents one **StockCode**, with its ABC and XYZ classes and key statistics."
        )
        st.dataframe(
            df_result[[
                "StockCode", "Description", "total_revenue",
                "ABC_Class", "XYZ_Class", "stock_class",
                "average_sales", "std_dev", "CV"
            ]].sort_values("total_revenue", ascending=False),
            use_container_width=True
        )

        # Download ABC–XYZ result
        st.download_button(
            "⬇ Download ABC–XYZ Classification as CSV",
            data=df_result.to_csv(index=False).encode("utf-8"),
            file_name="abc_xyz_classification.csv",
            mime="text/csv",
            help="Exports the ABC–XYZ result table."
        )

        st.markdown("#### Stock Class Distribution (AX, BY, CZ, etc.)")
        class_counts = df_result["stock_class"].value_counts().reset_index()
        class_counts.columns = ["stock_class", "count"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(class_counts["stock_class"], class_counts["count"])
        ax.set_title("Number of SKUs in Each ABC–XYZ Class")
        ax.set_xlabel("Stock Class")
        ax.set_ylabel("Count of SKUs")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # ABC–XYZ matrix as heatmap
        st.markdown("#### ABC–XYZ Matrix Heatmap")
        matrix = (
            df_result
            .groupby(["ABC_Class", "XYZ_Class"])["StockCode"]
            .nunique()
            .reset_index(name="sku_count")
        )
        matrix_pivot = matrix.pivot(index="ABC_Class", columns="XYZ_Class", values="sku_count").fillna(0)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix_pivot.values, cmap="Oranges")

        ax.set_xticks(np.arange(len(matrix_pivot.columns)))
        ax.set_yticks(np.arange(len(matrix_pivot.index)))
        ax.set_xticklabels(matrix_pivot.columns)
        ax.set_yticklabels(matrix_pivot.index)
        ax.set_xlabel("XYZ Class")
        ax.set_ylabel("ABC Class")
        ax.set_title("ABC–XYZ Matrix (Number of SKUs)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # annotate cells
        for i in range(len(matrix_pivot.index)):
            for j in range(len(matrix_pivot.columns)):
                ax.text(
                    j, i, int(matrix_pivot.iloc[i, j]),
                    ha="center", va="center", color="black", fontsize=9
                )

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Filter by Stock Class")
        selected_stock_class = st.selectbox(
            "Select a stock class (e.g. AX, BY, CZ):",
            options=sorted(df_result["stock_class"].unique())
        )
        filtered_df = df_result[df_result["stock_class"] == selected_stock_class]

        st.write(f"SKUs in class **{selected_stock_class}**:")
        st.dataframe(
            filtered_df[[
                "StockCode", "Description",
                "ABC_Class", "XYZ_Class",
                "total_revenue", "average_sales", "std_dev", "CV"
            ]].sort_values("total_revenue", ascending=False),
            use_container_width=True
        )

    else:
        st.warning("Not enough data to perform ABC–XYZ analysis with current filters.")

# -----------------------------
# TAB 4: RAW DATA
# -----------------------------
with tab4:
    st.subheader("🧾 Raw Transaction Data (Filtered)")
    st.write("This is the transaction-level data after applying the current filters.")

    max_rows = len(df_filtered)
    n_rows = st.slider(
        "Number of rows to display",
        min_value=10,
        max_value=max_rows if max_rows > 10 else 10,
        value=min(100, max_rows),
        step=10
    )

    st.dataframe(df_filtered.head(n_rows), use_container_width=True)
