import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Banggood Price Analysis", layout="wide")

# Load Data
df = pd.read_csv("banggood_cleaned.csv")

st.title("📊 Analysis 1: Price Distribution per Category")
st.write("This dashboard analyzes how product prices vary across categories on Banggood.")

# --- STATISTICS TABLE ---
st.subheader("📌 Price Statistics by Category")

price_stats = df.groupby('category')['price_numeric'].agg([
    ('Count', 'count'),
    ('Mean', 'mean'),
    ('Median', 'median'),
    ('Min', 'min'),
    ('Max', 'max'),
    ('Std Dev', 'std')
]).round(2)

st.dataframe(price_stats)

# --- KEY INSIGHTS ---
st.subheader("📌 Key Insights")

most_expensive = price_stats['Mean'].idxmax()
least_expensive = price_stats['Mean'].idxmin()

col1, col2 = st.columns(2)

with col1:
    st.success(f"💰 Most Expensive Category: {most_expensive}\n"
               f"Average Price: ${price_stats.loc[most_expensive, 'Mean']:.2f}")

with col2:
    st.info(f"💵 Least Expensive Category: {least_expensive}\n"
            f"Average Price: ${price_stats.loc[least_expensive, 'Mean']:.2f}")

# ---------------------------------------------------
# 📊 4 PLOTS IN ONE PAGE (2×2 GRID)
# ---------------------------------------------------
st.subheader("📊 Visual Price Analysis")


# ---------- ROW 1 ----------
colA, colB = st.columns(2)

# 1️⃣ Box Plot
with colA:
    fig, ax = plt.subplots(figsize=(8, 4))
    df.boxplot(column='price_numeric', by='category', ax=ax)
    ax.set_title("Price Distribution by Category (Box Plot)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Price ($)")
    plt.suptitle("")  # remove default title
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 2️⃣ Average Price Bar Chart
with colB:
    avg_prices = df.groupby('category')['price_numeric'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_prices.plot(kind="bar", ax=ax)
    ax.set_title("Average Price by Category")
    ax.set_ylabel("Average Price ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# ---------- ROW 2 ----------
colC, colD = st.columns(2)

# 3️⃣ Violin Plot
with colC:
    fig = plt.figure(figsize=(8, 4))
    sns.violinplot(data=df, x='category', y='price_numeric')
    plt.title("Price Distribution by Category (Violin Plot)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 4️⃣ Histogram
with colD:
    fig, ax = plt.subplots(figsize=(8, 4))
    for category in df['category'].unique():
        ax.hist(df[df['category'] == category]['price_numeric'],
                alpha=0.5, label=category)
    ax.set_title("Overlapping Price Histograms")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)


st.success("🎉 Analysis 1 Completed Successfully!")
st.title("📊 Analysis 2: Rating vs Price Correlation")

st.write("""
This analysis explores whether **expensive products receive higher ratings**.
It studies:
- Relationship between price and rating  
- Category-wise rating behavior  
- Statistical significance  
- Best performing price range  
""")

# --------------------
# CALCULATIONS
# --------------------

correlation = df['price_numeric'].corr(df['rating_numeric'])

st.subheader("📌 Overall Correlation")
st.metric("Correlation (Price vs Rating)", f"{correlation:.4f}")

# Interpretation
if correlation > 0.7:
    strength = "Strong Positive"
elif correlation > 0.3:
    strength = "Moderate Positive"
elif correlation > 0:
    strength = "Weak Positive"
elif correlation > -0.3:
    strength = "Weak Negative"
elif correlation > -0.7:
    strength = "Moderate Negative"
else:
    strength = "Strong Negative"

st.info(f"**Interpretation:** {strength} correlation")

# --------------------
# Correlation by Category
# --------------------

st.subheader("📌 Correlation by Category")

category_corr = df.groupby('category').apply(
    lambda x: x['price_numeric'].corr(x['rating_numeric'])
).round(4)

st.dataframe(category_corr)

# --------------------
# Price Range Binning
# --------------------
df['price_range'] = pd.cut(df['price_numeric'],
                           bins=5,
                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

range_stats = df.groupby('price_range')['rating_numeric'].agg(
    ['count', 'mean', 'min', 'max']
).round(2)

st.subheader("📌 Rating by Price Range")
st.dataframe(range_stats)

best_range = range_stats['mean'].idxmax()

st.success(f"⭐ **Best Rated Price Range:** {best_range}")

# ---------------------------------------------------
# 📊 4 PLOTS — 2×2 GRID
# ---------------------------------------------------

st.subheader("📊 Visual Analysis")

# ---------- ROW 1 ----------
col1, col2 = st.columns(2)

# 1️⃣ Scatter + Trend Line
with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df['price_numeric'], df['rating_numeric'], alpha=0.5)

    # Trend line
    z = np.polyfit(df['price_numeric'], df['rating_numeric'], 1)
    p = np.poly1d(z)
    ax.plot(df['price_numeric'], p(df['price_numeric']), "r--")

    ax.set_title("Rating vs Price (Scatter + Trend Line)")
    ax.set_xlabel("Price")
    ax.set_ylabel("Rating")

    st.pyplot(fig)

# 2️⃣ Category-wise Scatter
with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    for cat in df['category'].unique():
        ax.scatter(df[df['category'] == cat]['price_numeric'],
                   df[df['category'] == cat]['rating_numeric'],
                   label=cat, alpha=0.6)

    ax.set_title("Rating vs Price by Category")
    ax.set_xlabel("Price")
    ax.set_ylabel("Rating")
    ax.legend(fontsize=6)

    st.pyplot(fig)

# ---------- ROW 2 ----------
col3, col4 = st.columns(2)

# 3️⃣ Bar Chart - Avg Rating by Price Range
with col3:
    avg_rating_range = df.groupby('price_range')['rating_numeric'].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    avg_rating_range.plot(kind='bar', ax=ax)

    ax.set_title("Avg Rating by Price Range")
    ax.set_ylabel("Avg Rating")

    st.pyplot(fig)

# 4️⃣ Heatmap
with col4:
    corr = df[['price_numeric', 'rating_numeric', 'review_count', 'popularity_score']].corr()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

st.success("🎉 Analysis 2 Completed Successfully!")

# ======================================================
#                ANALYSIS 3 – TOP REVIEWED PRODUCTS
# ======================================================

st.title("📈 Analysis 3: Top Reviewed Products")

st.write("""
This analysis highlights **which products receive the most customer engagement**
based on review counts.

We explore:
- Overall review statistics  
- Top 10 most reviewed products  
- Most reviewed product in each category  
- Review distribution  
- Category-wise review insights  
- 6 visual plots  
""")

# ------------------------------
# Overall Review Statistics
# ------------------------------
st.subheader("📌 Overall Review Statistics")

colA, colB, colC, colD = st.columns(4)
colA.metric("Total Products", len(df))
colB.metric("Total Reviews", f"{df['review_count'].sum():,}")
colC.metric("Avg Reviews", f"{df['review_count'].mean():.2f}")
colD.metric("Median Reviews", f"{df['review_count'].median():.0f}")

colE, colF = st.columns(2)
colE.metric("Products with 0 Reviews", (df['review_count'] == 0).sum())
colF.metric("Products with Reviews", (df['review_count'] > 0).sum())

# ------------------------------
# Top 10 Most Reviewed Products
# ------------------------------
st.subheader("🏆 Top 10 Most Reviewed Products")

top_10 = df.nlargest(10, 'review_count')[[
    'name', 'category', 'price_numeric', 'rating_numeric', 'review_count'
]]

st.dataframe(top_10)

# ------------------------------
# Top Reviewed Product by Category
# ------------------------------
st.subheader("🏅 Top Reviewed Product in Each Category")

top_by_cat = []
for category in df['category'].unique():
    best = df[df['category'] == category].nlargest(1, 'review_count').iloc[0]
    top_by_cat.append([category, best['name'], best['review_count'],
                       best['rating_numeric'], best['price_numeric']])

top_cat_df = pd.DataFrame(top_by_cat,
                          columns=['Category', 'Product Name', 'Review Count',
                                   'Rating', 'Price'])

st.dataframe(top_cat_df)

# ------------------------------
# Review Count Distribution
# ------------------------------
df['review_category'] = pd.cut(df['review_count'],
                               bins=[0, 10, 50, 100, 500, float('inf')],
                               labels=['0-10', '11-50', '51-100', '101-500', '500+'])

review_dist = df['review_category'].value_counts().sort_index()

st.subheader("📊 Review Count Distribution")
st.dataframe(review_dist.to_frame("Number of Products"))

# ------------------------------
# Category-wise Review Statistics
# ------------------------------
st.subheader("📊 Review Statistics by Category")

category_stats = df.groupby('category')['review_count'].agg(
    Total_Products='count',
    Total_Reviews='sum',
    Avg_Reviews='mean',
    Max_Reviews='max'
).round(2)

st.dataframe(category_stats)

best_cat = category_stats['Total_Reviews'].idxmax()
st.success(f"🏆 Most Engaged Category: **{best_cat}** "
           f"({category_stats.loc[best_cat,'Total_Reviews']:,.0f} total reviews)")

# ======================================================
#                    6 CHARTS (3×2 GRID)
# ======================================================

st.subheader("📊 Visual Charts (6 Plots)")

# ROW 1
col1, col2 = st.columns(2)

# 1️⃣ Barh – Top 10 most reviewed
with col1:
    fig, ax = plt.subplots(figsize=(6,4))
    y = np.arange(len(top_10))
    colors = plt.cm.viridis(np.linspace(0,1,len(top_10)))

    ax.barh(y, top_10['review_count'], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([n[:30] + "..." for n in top_10['name']])
    ax.invert_yaxis()
    ax.set_title("Top 10 Most Reviewed Products")
    ax.set_xlabel("Review Count")
    st.pyplot(fig)

# 2️⃣ Histogram – Review distribution
with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df['review_count'], bins=50, alpha=0.7)
    ax.axvline(df['review_count'].mean(), linestyle="--", linewidth=2)
    ax.set_title("Review Count Distribution")
    ax.set_xlabel("Reviews")
    ax.set_ylabel("Number of Products")
    st.pyplot(fig)

# ROW 2
col3, col4 = st.columns(2)

# 3️⃣ Bar – Total reviews by category
with col3:
    fig, ax = plt.subplots(figsize=(6,4))
    data = df.groupby('category')['review_count'].sum().sort_values()
    ax.bar(data.index, data.values)
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Total Reviews by Category")
    st.pyplot(fig)

# 4️⃣ Bar – Average reviews per category
with col4:
    fig, ax = plt.subplots(figsize=(6,4))
    data = df.groupby('category')['review_count'].mean().sort_values()
    ax.bar(data.index, data.values)
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Average Reviews per Product by Category")
    st.pyplot(fig)

# ROW 3
col5, col6 = st.columns(2)

# 5️⃣ Pie chart – Review range distribution
with col5:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(review_dist.values, labels=review_dist.index, autopct="%1.1f%%")
    ax.set_title("Product Distribution by Review Range")
    st.pyplot(fig)

# 6️⃣ Scatter – Reviews vs Rating
with col6:
    fig, ax = plt.subplots(figsize=(6,4))
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        ax.scatter(cat_data['review_count'], cat_data['rating_numeric'],
                   label=cat, alpha=0.6)
    ax.set_xlabel("Review Count")
    ax.set_ylabel("Rating")
    ax.set_title("Review Count vs Rating by Category")
    ax.legend(fontsize=6)
    st.pyplot(fig)

st.success("🎉 Analysis 3 Completed Successfully!")


# ======================================================
#                ANALYSIS 4 – BEST VALUE METRIC
# ======================================================

st.title("💎 Analysis 4: Best Value Metric per Category")

st.write("""
This analysis identifies **products that offer the best value for money**, by combining  
**rating (quality)** and **price (affordability)** into a single **Value Score (0–100)**.

### 🔢 Value Score Formula  
`Value Score = 0.6 × (Rating / Max Rating) + 0.4 × (1 - Normalized Price)`  
Higher score = **Better value** 🚀  
""")

# ------------------------------
# VALUE SCORE CALCULATION
# ------------------------------
max_rating = df['rating_numeric'].max()
max_price = df['price_numeric'].max()
min_price = df['price_numeric'].min()

df['value_score'] = (
    (df['rating_numeric'] / max_rating) * 0.6 +
    (1 - (df['price_numeric'] - min_price) / (max_price - min_price)) * 0.4
) * 100

st.success("Value Score calculated successfully (0–100 scale).")

# ------------------------------
# TOP 10 BEST VALUE PRODUCTS
# ------------------------------
st.subheader("🏆 Top 10 Best Value Products")

top_value = df.nlargest(10, 'value_score')[[
    'name', 'category', 'price_numeric', 'rating_numeric',
    'review_count', 'value_score'
]]

st.dataframe(top_value)

# ------------------------------
# BEST VALUE PRODUCT PER CATEGORY
# ------------------------------
st.subheader("🥇 Best Value Product in Each Category")

best_value_rows = []
for category in df['category'].unique():
    best = df[df['category'] == category].nlargest(1, 'value_score').iloc[0]
    best_value_rows.append([
        category,
        best['name'],
        best['price_numeric'],
        best['rating_numeric'],
        best['review_count'],
        best['value_score']
    ])

best_value_cat_df = pd.DataFrame(
    best_value_rows,
    columns=['Category','Product','Price','Rating','Reviews','Value Score']
)

st.dataframe(best_value_cat_df)

# ------------------------------
# VALUE SCORE STATISTICS BY CATEGORY
# ------------------------------
st.subheader("📊 Value Score Statistics by Category")

value_stats = df.groupby('category')['value_score'].agg(
    Count='count',
    Mean='mean',
    Max='max',
    Min='min'
).round(2)

st.dataframe(value_stats)

# ------------------------------
# PRICE CATEGORY VS VALUE
# ------------------------------
if 'price_category' in df.columns:
    st.subheader("💰 Price Category vs Value Score")

    price_value_stats = df.groupby('price_category')['value_score'].agg(
        Products='count',
        Avg_Value_Score='mean'
    ).round(2)

    st.dataframe(price_value_stats)

# ======================================================
#                     6 CHARTS (3×2 GRID)
# ======================================================

st.subheader("📈 Visual Charts (6 Plots)")

# ROW 1
col1, col2 = st.columns(2)

# 1️⃣ Top 10 Best Value (Barh)
with col1:
    fig, ax = plt.subplots(figsize=(6,4))
    top10 = df.nlargest(10, 'value_score')
    y = np.arange(len(top10))
    ax.barh(y, top10['value_score'], color=plt.cm.RdYlGn(np.linspace(0.3,1,10)))
    ax.set_yticks(y)
    ax.set_yticklabels([name[:25]+"..." for name in top10['name']])
    ax.invert_yaxis()
    ax.set_title("Top 10 Best Value Products")
    ax.set_xlabel("Value Score")
    st.pyplot(fig)

# 2️⃣ Best Value Per Category (Barh)
with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    sorted_cat = best_value_cat_df.sort_values("Value Score", ascending=False)
    y = np.arange(len(sorted_cat))
    ax.barh(y, sorted_cat['Value Score'], color=plt.cm.viridis(np.linspace(0,1,len(sorted_cat))))
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_cat['Category'])
    ax.invert_yaxis()
    ax.set_title("Best Value Product per Category")
    ax.set_xlabel("Value Score")
    st.pyplot(fig)

# ROW 2
col3, col4 = st.columns(2)

# 3️⃣ Value Score Distribution
with col3:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df['value_score'], bins=40, alpha=0.7)
    ax.axvline(df['value_score'].mean(), linestyle='--', linewidth=2)
    ax.set_title("Value Score Distribution")
    ax.set_xlabel("Value Score")
    ax.set_ylabel("Product Count")
    st.pyplot(fig)

# 4️⃣ Price vs Value Score Scatter
with col4:
    fig, ax = plt.subplots(figsize=(6,4))
    scatter = ax.scatter(
        df['price_numeric'],
        df['value_score'],
        c=df['rating_numeric'],
        cmap='RdYlGn',
        s=50,
        alpha=0.6
    )
    plt.colorbar(scatter, label="Rating")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Value Score")
    ax.set_title("Price vs Value Score (Colored by Rating)")
    st.pyplot(fig)

# ROW 3
col5, col6 = st.columns(2)

# 5️⃣ Avg Value Score by Category
with col5:
    fig, ax = plt.subplots(figsize=(6,4))
    avg_values = df.groupby('category')['value_score'].mean().sort_values()
    ax.bar(avg_values.index, avg_values.values)
    plt.xticks(rotation=45, ha='right')
    ax.set_title("Average Value Score by Category")
    ax.set_ylabel("Value Score")
    st.pyplot(fig)

# 6️⃣ Value Score by Price Category
if 'price_category' in df.columns:
    with col6:
        fig, ax = plt.subplots(figsize=(6,4))
        df.boxplot(column='value_score', by='price_category', ax=ax)
        ax.set_title("Value Score by Price Category")
        ax.set_ylabel("Value Score")
        plt.suptitle("")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

st.success("🎉 Analysis 4 Completed Successfully!")
