"""
SQLite database for financial data.

Seeds three tables from the original demo datasets at import time.
Exposes read-only query tools for the Deep Agent.
"""

import sqlite3
import os

import pandas as pd
from langchain_core.tools import tool

DB_PATH = os.path.join(os.path.dirname(__file__), "financial_data.db")

company_financials = pd.DataFrame({
    "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "Revenue": [2500000, 2750000, 3100000, 3450000],
    "COGS": [1500000, 1600000, 1750000, 1850000],
    "Gross_Profit": [1000000, 1150000, 1350000, 1600000],
    "Operating_Expenses": [600000, 650000, 700000, 750000],
    "Net_Income": [400000, 500000, 650000, 850000],
    "Gross_Margin_Pct": [40.0, 41.8, 43.5, 46.4],
    "Net_Margin_Pct": [16.0, 18.2, 21.0, 24.6],
    "Customers": [1200, 1350, 1520, 1750],
    "Employees": [45, 52, 58, 65],
})

saas_metrics = pd.DataFrame({
    "Month": ["Sep 2024", "Oct 2024", "Nov 2024", "Dec 2024"],
    "MRR": [95000, 112000, 135000, 165000],
    "ARR": [1140000, 1344000, 1620000, 1980000],
    "New_Customers": [45, 62, 78, 95],
    "Churned_Customers": [8, 6, 7, 5],
    "Net_New_MRR": [12000, 17000, 23000, 30000],
    "Churn_Rate_Pct": [3.5, 2.8, 2.4, 1.9],
    "CAC": [520, 485, 450, 410],
    "LTV": [4800, 5200, 5800, 6500],
    "LTV_CAC_Ratio": [9.2, 10.7, 12.9, 15.9],
})

ecommerce_data = pd.DataFrame({
    "Category": ["Electronics", "Apparel", "Home & Garden", "Sports", "Beauty"],
    "Revenue": [1250000, 890000, 650000, 420000, 380000],
    "Orders": [8500, 12000, 5200, 3800, 6500],
    "Avg_Order_Value": [147, 74, 125, 111, 58],
    "Return_Rate_Pct": [8.5, 15.2, 4.3, 6.8, 3.2],
    "Profit_Margin_Pct": [18.5, 42.0, 35.0, 28.0, 55.0],
    "YoY_Growth_Pct": [12.0, 25.0, 8.0, 35.0, 45.0],
})


# Deliberately overlapping revenue data — same quarters, different cuts
regional_revenue = pd.DataFrame({
    "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "Region": ["North America", "North America", "North America", "North America"],
    "Revenue": [1500000, 1650000, 1860000, 2070000],
    "Customers": [720, 810, 912, 1050],
    "Net_Income": [240000, 300000, 390000, 510000],
    "Growth_Pct": [15.0, 10.0, 12.7, 11.3],
})

regional_revenue_emea = pd.DataFrame({
    "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "Region": ["EMEA", "EMEA", "EMEA", "EMEA"],
    "Revenue": [650000, 715000, 806000, 897000],
    "Customers": [310, 348, 394, 455],
    "Net_Income": [104000, 129000, 169000, 221000],
    "Growth_Pct": [18.0, 10.0, 12.7, 11.3],
})

regional_revenue_apac = pd.DataFrame({
    "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "Region": ["APAC", "APAC", "APAC", "APAC"],
    "Revenue": [350000, 385000, 434000, 483000],
    "Customers": [170, 192, 214, 245],
    "Net_Income": [56000, 71000, 91000, 119000],
    "Growth_Pct": [22.0, 10.0, 12.7, 11.3],
})

# Combine into one table
regional_revenue_all = pd.concat([regional_revenue, regional_revenue_emea, regional_revenue_apac], ignore_index=True)

# Product line data — overlaps with ecommerce categories
product_lines = pd.DataFrame({
    "Product": ["Electronics Pro", "Electronics Basic", "Apparel Premium", "Apparel Standard",
                "Home Essentials", "Sports Gear", "Beauty Plus", "Beauty Basics"],
    "Category": ["Electronics", "Electronics", "Apparel", "Apparel",
                 "Home & Garden", "Sports", "Beauty", "Beauty"],
    "Revenue": [820000, 430000, 520000, 370000, 650000, 420000, 210000, 170000],
    "Units_Sold": [3200, 5300, 4800, 7200, 5200, 3800, 3100, 3400],
    "Profit_Margin_Pct": [22.0, 12.5, 48.0, 34.0, 35.0, 28.0, 62.0, 45.0],
    "Return_Rate_Pct": [6.2, 12.8, 11.5, 20.1, 4.3, 6.8, 2.1, 4.8],
    "Customer_Rating": [4.5, 3.8, 4.7, 4.1, 4.3, 4.4, 4.8, 4.2],
})

# Headcount data — overlaps with company_financials employee counts
headcount = pd.DataFrame({
    "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "Engineering": [18, 22, 24, 28],
    "Sales": [12, 14, 16, 18],
    "Marketing": [6, 7, 8, 9],
    "Support": [5, 5, 6, 6],
    "Operations": [4, 4, 4, 4],
    "Total": [45, 52, 58, 65],
    "Revenue_Per_Employee": [55556, 52885, 53448, 53077],
    "Hiring_Budget_Remaining": [120000, 85000, 45000, 15000],
})

# Monthly burn rate — overlaps with SaaS MRR data
monthly_financials = pd.DataFrame({
    "Month": ["Sep 2024", "Oct 2024", "Nov 2024", "Dec 2024"],
    "Revenue": [95000, 112000, 135000, 165000],
    "Payroll": [62000, 68000, 72000, 78000],
    "Infrastructure": [12000, 14000, 15000, 16000],
    "Marketing_Spend": [18000, 22000, 28000, 35000],
    "Other_Costs": [8000, 9000, 10000, 11000],
    "Net_Burn": [5000, 1000, -10000, -25000],
    "Cash_Balance": [850000, 851000, 861000, 886000],
    "Runway_Months": [170.0, 851.0, -86.1, -35.4],
})

# Customer cohort data — another angle on churn
customer_cohorts = pd.DataFrame({
    "Cohort": ["Jan 2024", "Mar 2024", "Jun 2024", "Sep 2024"],
    "Initial_Customers": [120, 95, 150, 180],
    "Month_1_Retained": [108, 88, 138, 167],
    "Month_3_Retained": [92, 76, 120, 155],
    "Month_6_Retained": [78, 64, 105, None],
    "Avg_Revenue_Per_Customer": [85, 92, 78, 95],
    "Expansion_Revenue_Pct": [12.0, 15.0, 8.0, 18.0],
})


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    company_financials.to_sql("company_financials", conn, if_exists="replace", index=False)
    saas_metrics.to_sql("saas_metrics", conn, if_exists="replace", index=False)
    ecommerce_data.to_sql("ecommerce_data", conn, if_exists="replace", index=False)
    regional_revenue_all.to_sql("regional_revenue", conn, if_exists="replace", index=False)
    product_lines.to_sql("product_lines", conn, if_exists="replace", index=False)
    headcount.to_sql("headcount", conn, if_exists="replace", index=False)
    monthly_financials.to_sql("monthly_financials", conn, if_exists="replace", index=False)
    customer_cohorts.to_sql("customer_cohorts", conn, if_exists="replace", index=False)
    conn.close()


_init_db()


FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH"}


@tool
def list_tables() -> str:
    """List all available tables in the financial database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "Available tables: " + ", ".join(tables)


@tool
def describe_table(table_name: str) -> str:
    """Describe a table's columns and show the first 3 rows as a sample."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        if not columns:
            return f"Table '{table_name}' not found."
        col_info = "\n".join(f"  {c[1]} ({c[2]})" for c in columns)
        sample = conn.execute(f"SELECT * FROM '{table_name}' LIMIT 3")
        col_names = [desc[0] for desc in sample.description]
        rows = sample.fetchall()
        sample_str = "\n".join(
            "  " + " | ".join(str(v) for v in row) for row in rows
        )
        return f"Table: {table_name}\n\nColumns:\n{col_info}\n\nSample rows ({', '.join(col_names)}):\n{sample_str}"
    finally:
        conn.close()


@tool
def query_financials(sql: str) -> str:
    """Execute a read-only SQL query against the financial database. Returns formatted results."""
    upper = sql.upper().strip()
    for kw in FORBIDDEN:
        if kw in upper.split():
            return f"Error: {kw} statements are not allowed. Read-only queries only."
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(sql)
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        if not rows:
            return "Query returned no results."
        header = " | ".join(col_names)
        divider = "-" * len(header)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        return f"{header}\n{divider}\n{body}"
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        conn.close()
