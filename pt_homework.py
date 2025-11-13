import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="B2B Intelligent Dashboard", layout="wide")

st.title("📊 B2B Intelligent Sales Dashboard")
st.write(
    "This dashboard is built with *Streamlit* using the B2B Transaction dataset. "
    "It includes KPIs, interactive visuals and an *ABC–XYZ stock classification* analysis."
)

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data(uploaded_file):
    """Read Excel from uploader (or local file as fallback)."""
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        # Fallback for running locally (teacher can keep the file in the same folder)
        df = pd.read_excel("B2B_Transaction_Data.xlsx")
    return df

st.sidebar.header("🔁 Data & Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload *B2B_Transaction_Data.xlsx*", type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.sidebar.info(
        "You can upload the homework dataset here.\n\n"
        "If you are running this locally and the file is in the same folder "
        "as this script, the app will try to load it automatically."
    )

# Try loading data (will crash if file is really missing everywhere, which is okay for homework)
try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error("❌ Data could not be loaded. Please upload the Excel file.")
    st.stop()

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
# Basic cleaning & feature engineering

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

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = max_date

st.sidebar.subheader("🔎 Other Filters")

categories = st.sidebar.multiselect(
    "Category", options=sorted(df["Category"].dropna().unique()),
    default=sorted(df["Category"].dropna().unique())
)

cities = st.sidebar.multiselect(
    "City", options=sorted(df["City"].dropna().unique()),
    default=sorted(df["City"].dropna().unique())
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
    f"Date range: *{start_date}* → *{end_date}*  |  "
    f"Categories: *{len(categories)} selected*  |  "
    f"Cities: *{len(cities)} selected*  |  "
    f"Rows after filtering: *{len(df_filtered):,}*"
)

# -----------------------------
# TABS (LIKE HOCANIN ÖRNEKLERİ)
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
    col1.metric("Total Revenue", f"{total_revenue:,.2f} ₺")
    col2.metric("Total Quantity", f"{total_quantity:,.0f}")
    col3.metric("Number of Invoices", f"{total_invoices:,}")
    col4.metric("Number of Customers", f"{total_customers:,}")

    st.divider()

    # Layout for charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Revenue by Category")
        revenue_by_cat = (
            df_filtered.groupby("Category")["SalesRevenue"]
            .sum()
            .reset_index()
            .sort_values("SalesRevenue", ascending=False)
        )
        fig_cat = px.bar(
            revenue_by_cat,
            x="Category",
            y="SalesRevenue",
            title="Total Revenue by Category",
            text_auto=".2s"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        st.markdown("#### Revenue by City (Top 10)")
        revenue_by_city = (
            df_filtered.groupby("City")["SalesRevenue"]
            .sum()
            .reset_index()
            .sort_values("SalesRevenue", ascending=False)
            .head(10)
        )
        fig_city = px.bar(
            revenue_by_city,
            x="City",
            y="SalesRevenue",
            title="Top 10 Cities by Revenue",
            text_auto=".2s"
        )
        st.plotly_chart(fig_city, use_container_width=True)

    st.markdown("#### Monthly Revenue Trend")
    revenue_over_time = (
        df_filtered.groupby("Month")["SalesRevenue"]
        .sum()
        .reset_index()
        .sort_values("Month")
    )
    fig_month = px.line(
        revenue_over_time,
        x="Month",
        y="SalesRevenue",
        markers=True,
        title="Monthly Revenue Trend"
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("#### Top 15 Products by Revenue")
    top_products = (
        df_filtered.groupby(["StockCode", "Description"])["SalesRevenue"]
        .sum()
        .reset_index()
        .sort_values("SalesRevenue", ascending=False)
        .head(15)
    )
    fig_prod = px.bar(
        top_products,
        x="Description",
            y="SalesRevenue",
        title="Top 15 Products by Revenue",
        text_auto=".2s"
    )
    fig_prod.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_prod, use_container_width=True)

# -----------------------------
# TAB 2: DESCRIPTIVE STATISTICS
# -----------------------------
with tab2:
    st.subheader("📈 Descriptive Statistics")

    st.markdown("#### Numeric Columns Summary")
    st.write(df_filtered[["Quantity", "NetPrice", "UnitPrice", "SalesRevenue"]].describe())

    st.markdown("#### Distribution of a Numeric Variable")
    numeric_col = st.selectbox(
        "Select numeric column",
        options=["Quantity", "NetPrice", "UnitPrice", "SalesRevenue"],
        index=3
    )

    fig_hist = px.histogram(
        df_filtered,
        x=numeric_col,
        nbins=40,
        title=f"Distribution of {numeric_col}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Box Plot for Outlier Detection")
    fig_box = px.box(
        df_filtered,
        y=numeric_col,
        title=f"Box Plot of {numeric_col}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# TAB 3: ABC–XYZ ANALYSIS
# -----------------------------
with tab3:
    st.subheader("📦 ABC–XYZ Stock Classification")

    st.write(
        "ABC–XYZ analysis groups SKUs (StockCode) based on *revenue importance (ABC)* "
        "and *demand variability (XYZ)*. This helps inventory and supply chain decisions."
    )

    # --- Hocanın notebook mantığına paralel ---
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
            if x <= 0.5:
                return "X"
            elif x > 0.5 and x <= 1:
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
        def abc_classification(x):
            if x > 0 and x <= 0.80:
                return "A"
            elif x > 0.80 and x <= 0.95:
                return "B"
            else:
                return "C"

        df_4["ABC_Class"] = df_4["sku_percent"].apply(abc_classification)

        # 8) Merge ABC & XYZ info
        df_3_small = df_3[["StockCode", "total_sales", "average_sales", "std_dev", "CV", "XYZ_Class"]]
        df_4_small = df_4[["StockCode", "total_revenue", "sku_percent", "ABC_Class"]]

        df_final = df_4_small.merge(df_3_small, on="StockCode", how="left")

        # 9) Bring Description from original df
        df_desc = df_filtered[["StockCode", "Description"]].drop_duplicates()
        df_merge = df_final.merge(df_desc, on="StockCode", how="left")

        # 10) Remove duplicates and create final stock class
        df_result = df_merge.drop_duplicates().copy()
        df_result["stock_class"] = df_result["ABC_Class"].astype(str) + df_result["XYZ_Class"].astype(str)

        st.markdown("#### ABC–XYZ Summary Table")
        st.write(
            "Each row represents one *StockCode*, with its ABC and XYZ classes and key statistics."
        )
        st.dataframe(
            df_result[[
                "StockCode", "Description", "total_revenue",
                "ABC_Class", "XYZ_Class", "stock_class",
                "average_sales", "std_dev", "CV"
            ]].sort_values("total_revenue", ascending=False)
        )

        st.markdown("#### Stock Class Distribution (AX, BY, CZ, etc.)")
        class_counts = df_result["stock_class"].value_counts().reset_index()
        class_counts.columns = ["stock_class", "count"]

        fig_classes = px.bar(
            class_counts,
            x="stock_class",
            y="count",
            title="Number of SKUs in Each ABC–XYZ Class",
            text_auto=True
        )
        st.plotly_chart(fig_classes, use_container_width=True)

        st.markdown("#### Filter by Stock Class")
        selected_stock_class = st.selectbox(
            "Select a stock class (e.g. AX, BY, CZ)",
            options=sorted(df_result["stock_class"].unique())
        )
        filtered_df = df_result[df_result["stock_class"] == selected_stock_class]

        st.write(f"SKUs in class *{selected_stock_class}*:")
        st.dataframe(filtered_df[[
            "StockCode", "Description",
            "ABC_Class", "XYZ_Class",
            "total_revenue", "average_sales", "std_dev", "CV"
        ]])

    else:
        st.warning("Not enough data to perform ABC–XYZ analysis with current filters.")

# -----------------------------
# TAB 4: RAW DATA
# -----------------------------
with tab4:
    st.subheader("🧾 Raw Transaction Data (Filtered)")
    st.write("This is the filtered transaction-level data used in the dashboard.")
    st.dataframe(df_filtered)
