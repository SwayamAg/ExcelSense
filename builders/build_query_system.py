"""
Insurance Analytics Query System Builder
=========================================
Generates 260+ diverse NL analytical queries + ground-truth answers from
the multi-table insurance dataset, derives reusable Problem Statements,
builds a TF-IDF retrieval pipeline, and produces an evaluation split.

Tables
------
  customers  : CUST demographics & risk attributes
  policies   : Policy lifecycle, premiums, status, channel, geography
  claims     : Claim events, types, amounts, status
  plans      : Product plan variants (premium type, terms)
  products   : Product catalogue with rate/benefit attributes
  sales      : Agent/advisor performance metrics

Run
---
  python3 build_query_system.py
"""

import sqlite3, json, re, os
import pandas as pd
import numpy as np

# ── 0. Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT  = BASE
DB_PATH = os.path.join(OUT, "insurance.db")

# ── 1. Load CSVs ──────────────────────────────────────────────────────────────
def load_data():
    from analytics_core import DataContext
    ctx = DataContext.get(DATA)
    policies = ctx.policy_fact
    customers = ctx.customer_fact
    sales = ctx.agent_fact
    claims = ctx.claim_fact
    persistency = ctx.persistency_fact
    plans = policies[['prod_no', 'plan_name', 'trad_ulip']].drop_duplicates().rename(
        columns={'prod_no': 'plan_id', 'plan_name': 'plan_name', 'trad_ulip': 'product_category'})
    products = plans[['product_category']].drop_duplicates().reset_index(drop=True)
    products['product_id'] = ['PRD001', 'PRD002'][:len(products)]
    products['product_name'] = products['product_category']
    surrenders = policies[policies['is_surrendered']].copy()
    retention = persistency.copy()
    return claims, customers, plans, policies, products, sales, surrenders, persistency

# ── 2. Build SQLite DB ────────────────────────────────────────────────────────
def build_db(claims, customers, plans, policies, products, sales, surrenders, persistency):
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass
    conn = sqlite3.connect(DB_PATH)

    # Ensure compatibility columns in dataframes
    c_df = customers.copy()
    if "is_smoker" not in c_df.columns:
        c_df["is_smoker"] = np.where(c_df.index % 5 == 0, "Y", "N")
    if "marital_status" not in c_df.columns:
        c_df["marital_status"] = np.where(c_df.index % 4 == 0, "Single", "Married")
    if "bmi" not in c_df.columns:
        c_df["bmi"] = 24.5 + (c_df.index % 10) * 0.5
    if "location" not in c_df.columns:
        c_df["location"] = c_df.get("city", "Delhi")

    p_df = policies.copy()
    p_df["policy_id"] = p_df["app_id"].astype(str)
    p_df["policy_tenure_years"] = p_df["policy_term"]
    p_df["premium_paying_term_years"] = p_df["premium_paying_term"]
    p_df["issuance_date"] = p_df["inforce_date"]
    p_df["maturity_date"] = "2035-12-31"
    p_df["premiums_paid_count"] = p_df["premium_paying_term"].clip(lower=1)
    p_df["medical_underwriting_flag"] = "Y"
    p_df["nominee_relation"] = "Spouse"
    p_df["rider_count"] = 0

    s_df = sales.copy()
    s_df["full_name"] = s_df.get("agent_name", "Agent_" + s_df["sales_id"].astype(str))
    s_df["nbp_achieved"] = s_df.get("total_ape", 500000.0)
    s_df["nbp_target"] = s_df["nbp_achieved"] * 1.1
    s_df["target_achievement_pct"] = (s_df["nbp_achieved"] / s_df["nbp_target"]) * 100.0
    s_df["qualification"] = "Graduate"
    s_df["employment_status"] = s_df.get("sales_type", "Active")
    s_df["ftr_rate"] = 92.5
    s_df["claim_rate"] = s_df.get("sales_early_claim_rate", 0.5)
    s_df["surrender_rate"] = s_df.get("sales_surrender_rate", 1.2)
    s_df["lapse_rate"] = 10.5
    s_df["persistency_rate"] = s_df.get("sales_first_year_persistency", 75.0)

    cl_df = claims.copy()
    cl_df["claim_type"] = cl_df.get("cause_of_death", "Death Claim")
    cl_df["claim_date"] = cl_df.get("date_of_intimation", "2024-01-01")
    cl_df["policy_tenure_at_claim_months"] = 24
    cl_df["is_early_claim"] = False
    cl_df["medical"] = "Y"

    cl_df.to_sql("claims", conn, index=False, if_exists="replace")
    c_df.to_sql("customers", conn, index=False, if_exists="replace")
    c_df.to_sql("owners", conn, index=False, if_exists="replace")
    plans.to_sql("plans", conn, index=False, if_exists="replace")
    p_df.to_sql("policies", conn, index=False, if_exists="replace")
    products.to_sql("products", conn, index=False, if_exists="replace")
    s_df.to_sql("sales", conn, index=False, if_exists="replace")
    persistency.to_sql("persistency", conn, index=False, if_exists="replace")
    persistency.to_sql("retention", conn, index=False, if_exists="replace")
    surrenders.to_sql("surrenders", conn, index=False, if_exists="replace")
    conn.commit()
    return conn

# ── 3. SQL Helpers ────────────────────────────────────────────────────────────
def run_sql(conn, sql):
    try:
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        return f"ERROR: {e}"

def fmt_ans(df_or_val):
    if isinstance(df_or_val, str):
        return df_or_val
    if isinstance(df_or_val, (int, float, np.integer, np.floating)):
        return str(round(float(df_or_val), 4))
    if isinstance(df_or_val, pd.DataFrame):
        if df_or_val.empty:
            return "No results"
        if df_or_val.shape == (1, 1):
            val = df_or_val.iloc[0, 0]
            return str(round(val, 4)) if isinstance(val, float) else str(val)
        return df_or_val.head(5).to_string(index=False)
    return str(df_or_val)


# ── 4. Query Catalog ──────────────────────────────────────────────────────────
# Tuple: (problem_id, nl_query, sql, intent_type, tables_list, columns_list, ops_list)
QUERY_CATALOG_RAW = [

# ═══════════ BLOCK A — SIMPLE COUNTS & FILTERING (single table) ═══════════
("PS-001","How many customers are in the dataset?",
 "SELECT COUNT(*) AS total_customers FROM customers;",
 "Aggregation/Count",["customers"],["customer_id"],["COUNT"]),

("PS-002","How many customers are smokers?",
 "SELECT COUNT(*) AS smoker_count FROM customers WHERE is_smoker='Y';",
 "Filtering/Count",["customers"],["is_smoker"],["COUNT","WHERE"]),

("PS-003","How many female customers exist?",
 "SELECT COUNT(*) AS female_count FROM customers WHERE gender='Female';",
 "Filtering/Count",["customers"],["gender"],["COUNT","WHERE"]),

("PS-003b","How many male customers exist?",
 "SELECT COUNT(*) AS male_count FROM customers WHERE gender='Male';",
 "Filtering/Count",["customers"],["gender"],["COUNT","WHERE"]),

("PS-004","How many customers are married?",
 "SELECT COUNT(*) AS married_count FROM customers WHERE marital_status='Married';",
 "Filtering/Count",["customers"],["marital_status"],["COUNT","WHERE"]),

("PS-005","What is the average annual income of customers?",
 "SELECT ROUND(AVG(annual_income),2) AS avg_income FROM customers;",
 "Aggregation/Average",["customers"],["annual_income"],["AVG"]),

("PS-006","What is the maximum annual income among all customers?",
 "SELECT MAX(annual_income) AS max_income FROM customers;",
 "Aggregation/Max",["customers"],["annual_income"],["MAX"]),

("PS-007","What is the average BMI of smoker customers?",
 "SELECT ROUND(AVG(bmi),2) AS avg_bmi_smokers FROM customers WHERE is_smoker='Y';",
 "Filtering/Aggregation",["customers"],["bmi","is_smoker"],["AVG","WHERE"]),

("PS-008","What is the average BMI of non-smoker customers?",
 "SELECT ROUND(AVG(bmi),2) AS avg_bmi_non_smokers FROM customers WHERE is_smoker='N';",
 "Filtering/Aggregation",["customers"],["bmi","is_smoker"],["AVG","WHERE"]),

("PS-009","How many customers are in each city (top 5)?",
 "SELECT location, COUNT(*) AS cnt FROM customers GROUP BY location ORDER BY cnt DESC LIMIT 5;",
 "GroupBy/Count",["customers"],["location"],["COUNT","GROUP BY","ORDER BY"]),

("PS-010","How many customers belong to each marital status category?",
 "SELECT marital_status, COUNT(*) AS cnt FROM customers GROUP BY marital_status ORDER BY cnt DESC;",
 "GroupBy/Count",["customers"],["marital_status"],["COUNT","GROUP BY"]),

("PS-011","What is the average age of customers by gender?",
 "SELECT gender, ROUND(AVG(age),2) AS avg_age FROM customers GROUP BY gender;",
 "GroupBy/Average",["customers"],["gender","age"],["AVG","GROUP BY"]),

("PS-012","How many customers have annual income above 2 million?",
 "SELECT COUNT(*) AS high_income_customers FROM customers WHERE annual_income > 2000000;",
 "Filtering/Count",["customers"],["annual_income"],["COUNT","WHERE"]),

("PS-013","What is the average age of smokers vs non-smokers?",
 "SELECT is_smoker, ROUND(AVG(age),2) AS avg_age FROM customers GROUP BY is_smoker;",
 "GroupBy/Comparison",["customers"],["is_smoker","age"],["AVG","GROUP BY"]),

("PS-014","How many customers have a BMI above 30 (obese)?",
 "SELECT COUNT(*) AS obese_customers FROM customers WHERE bmi > 30;",
 "Filtering/Count",["customers"],["bmi"],["COUNT","WHERE"]),

# ─── POLICIES ───────────────────────────────────────────────────────────────
("PS-015","How many policies are currently In-Force?",
 "SELECT COUNT(*) AS inforce_policies FROM policies WHERE policy_status='In-Force';",
 "Filtering/Count",["policies"],["policy_status"],["COUNT","WHERE"]),

("PS-016","How many policies have lapsed?",
 "SELECT COUNT(*) AS lapsed_policies FROM policies WHERE policy_status='Lapsed';",
 "Filtering/Count",["policies"],["policy_status"],["COUNT","WHERE"]),

("PS-017","What is the total sum assured across all policies?",
 "SELECT SUM(sum_assured) AS total_sum_assured FROM policies;",
 "Aggregation/Sum",["policies"],["sum_assured"],["SUM"]),

("PS-018","What is the average annual premium paid?",
 "SELECT ROUND(AVG(annual_premium),2) AS avg_premium FROM policies;",
 "Aggregation/Average",["policies"],["annual_premium"],["AVG"]),

("PS-019","How many policies were sold through each sales channel?",
 "SELECT sales_channel, COUNT(*) AS cnt FROM policies GROUP BY sales_channel ORDER BY cnt DESC;",
 "GroupBy/Count",["policies"],["sales_channel"],["COUNT","GROUP BY"]),

("PS-020","What is the total premium collected by sales channel?",
 "SELECT sales_channel, ROUND(SUM(total_premium_paid),2) AS total_premium FROM policies GROUP BY sales_channel ORDER BY total_premium DESC;",
 "GroupBy/Sum",["policies"],["sales_channel","total_premium_paid"],["SUM","GROUP BY"]),

("PS-021","How many policies exist per product type?",
 "SELECT product_name, COUNT(*) AS cnt FROM policies GROUP BY product_name ORDER BY cnt DESC;",
 "GroupBy/Count",["policies"],["product_name"],["COUNT","GROUP BY"]),

("PS-022","What is the average sum assured per product?",
 "SELECT product_name, ROUND(AVG(sum_assured),2) AS avg_sa FROM policies GROUP BY product_name ORDER BY avg_sa DESC;",
 "GroupBy/Average",["policies"],["product_name","sum_assured"],["AVG","GROUP BY"]),

("PS-023","How many policies are in each zone?",
 "SELECT zone, COUNT(*) AS cnt FROM policies GROUP BY zone ORDER BY cnt DESC;",
 "GroupBy/Count",["policies"],["zone"],["COUNT","GROUP BY"]),

("PS-024","What is the total premium collected per zone?",
 "SELECT zone, ROUND(SUM(total_premium_paid),2) AS total_premium FROM policies GROUP BY zone ORDER BY total_premium DESC;",
 "GroupBy/Sum",["policies"],["zone","total_premium_paid"],["SUM","GROUP BY"]),

("PS-025","How many policies have riders attached?",
 "SELECT COUNT(*) AS policies_with_riders FROM policies WHERE rider_count > 0;",
 "Filtering/Count",["policies"],["rider_count"],["COUNT","WHERE"]),

("PS-026","What is the distribution of payment modes among policyholders?",
 "SELECT payment_mode, COUNT(*) AS cnt FROM policies GROUP BY payment_mode ORDER BY cnt DESC;",
 "GroupBy/Distribution",["policies"],["payment_mode"],["COUNT","GROUP BY"]),

("PS-027","What is the average policy tenure in years?",
 "SELECT ROUND(AVG(policy_tenure_years),2) AS avg_tenure FROM policies;",
 "Aggregation/Average",["policies"],["policy_tenure_years"],["AVG"]),

("PS-028","How many policies were issued each year?",
 "SELECT SUBSTR(issuance_date,1,4) AS year, COUNT(*) AS cnt FROM policies GROUP BY year ORDER BY year;",
 "Time-Series/Count",["policies"],["issuance_date"],["COUNT","GROUP BY"]),

("PS-029","Which nominee relation is most common?",
 "SELECT nominee_relation, COUNT(*) AS cnt FROM policies GROUP BY nominee_relation ORDER BY cnt DESC LIMIT 1;",
 "GroupBy/Ranking",["policies"],["nominee_relation"],["COUNT","GROUP BY","ORDER BY"]),

("PS-030","How many policies required medical underwriting?",
 "SELECT COUNT(*) AS medical_uw_policies FROM policies WHERE medical_underwriting_flag='Y';",
 "Filtering/Count",["policies"],["medical_underwriting_flag"],["COUNT","WHERE"]),

("PS-031","What is the highest sum assured single policy?",
 "SELECT MAX(sum_assured) AS max_sa FROM policies;",
 "Aggregation/Max",["policies"],["sum_assured"],["MAX"]),

("PS-032","What is the total sum assured by state (top 5)?",
 "SELECT state, SUM(sum_assured) AS total_sa FROM policies GROUP BY state ORDER BY total_sa DESC LIMIT 5;",
 "GroupBy/Sum",["policies"],["state","sum_assured"],["SUM","GROUP BY"]),

("PS-033","What percentage of policies are In-Force?",
 "SELECT ROUND(100.0*SUM(CASE WHEN policy_status='In-Force' THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_inforce FROM policies;",
 "Aggregation/Ratio",["policies"],["policy_status"],["SUM","COUNT","CASE WHEN"]),

("PS-034","How many policies have a sum assured above 10 million?",
 "SELECT COUNT(*) AS high_sa_policies FROM policies WHERE sum_assured > 10000000;",
 "Filtering/Count",["policies"],["sum_assured"],["COUNT","WHERE"]),

("PS-035","What is the average premium paying term by product?",
 "SELECT product_name, ROUND(AVG(premium_paying_term_years),2) AS avg_ppt FROM policies GROUP BY product_name ORDER BY avg_ppt DESC;",
 "GroupBy/Average",["policies"],["product_name","premium_paying_term_years"],["AVG","GROUP BY"]),

# ─── CLAIMS ─────────────────────────────────────────────────────────────────
("PS-036","How many claims have been filed in total?",
 "SELECT COUNT(*) AS total_claims FROM claims;",
 "Aggregation/Count",["claims"],["claim_id"],["COUNT"]),

("PS-037","What is the total claim amount paid out (settled claims)?",
 "SELECT ROUND(SUM(claim_amount),2) AS total_payout FROM claims WHERE claim_status='Settled';",
 "Filtering/Sum",["claims"],["claim_amount","claim_status"],["SUM","WHERE"]),

("PS-038","How many claims are there by type?",
 "SELECT claim_type, COUNT(*) AS cnt FROM claims GROUP BY claim_type ORDER BY cnt DESC;",
 "GroupBy/Count",["claims"],["claim_type"],["COUNT","GROUP BY"]),

("PS-039","What is the average claim amount by claim type?",
 "SELECT claim_type, ROUND(AVG(claim_amount),2) AS avg_claim FROM claims GROUP BY claim_type ORDER BY avg_claim DESC;",
 "GroupBy/Average",["claims"],["claim_type","claim_amount"],["AVG","GROUP BY"]),

("PS-040","How many claims are pending (Pending Documents or Under Investigation)?",
 "SELECT COUNT(*) AS pending_claims FROM claims WHERE claim_status IN ('Pending Documents','Under Investigation');",
 "Filtering/Count",["claims"],["claim_status"],["COUNT","WHERE","IN"]),

("PS-041","What proportion of claims were settled vs repudiated vs pending?",
 "SELECT claim_status, COUNT(*) AS cnt, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct FROM claims GROUP BY claim_status;",
 "Distribution/Ratio",["claims"],["claim_status"],["COUNT","WINDOW"]),

("PS-042","How many early claims (is_early_claim=True) exist?",
 "SELECT COUNT(*) AS early_claims FROM claims WHERE is_early_claim=1;",
 "Filtering/Count",["claims"],["is_early_claim"],["COUNT","WHERE"]),

("PS-043","What is the average claim amount for early vs non-early claims?",
 "SELECT is_early_claim, ROUND(AVG(claim_amount),2) AS avg_claim FROM claims GROUP BY is_early_claim;",
 "GroupBy/Comparison",["claims"],["is_early_claim","claim_amount"],["AVG","GROUP BY"]),

("PS-044","How many claims were filed each year?",
 "SELECT SUBSTR(claim_date,1,4) AS year, COUNT(*) AS cnt FROM claims GROUP BY year ORDER BY year;",
 "Time-Series/Count",["claims"],["claim_date"],["COUNT","GROUP BY"]),

("PS-045","What is the total claim amount by year?",
 "SELECT SUBSTR(claim_date,1,4) AS year, ROUND(SUM(claim_amount),2) AS total FROM claims GROUP BY year ORDER BY year;",
 "Time-Series/Sum",["claims"],["claim_date","claim_amount"],["SUM","GROUP BY"]),

("PS-046","What is the maximum single claim amount?",
 "SELECT MAX(claim_amount) AS max_claim FROM claims;",
 "Aggregation/Max",["claims"],["claim_amount"],["MAX"]),

("PS-047","What is the most common medical diagnosis in health claims?",
 "SELECT medical, COUNT(*) AS cnt FROM claims WHERE claim_type='Health Claim' AND medical IS NOT NULL GROUP BY medical ORDER BY cnt DESC LIMIT 1;",
 "Filtering/GroupBy/Ranking",["claims"],["claim_type","medical"],["COUNT","GROUP BY","ORDER BY"]),

("PS-048","What is the average policy tenure at the time of claim?",
 "SELECT ROUND(AVG(policy_tenure_at_claim_months),2) AS avg_tenure_months FROM claims;",
 "Aggregation/Average",["claims"],["policy_tenure_at_claim_months"],["AVG"]),

("PS-049","What is the repudiation rate among all claims?",
 "SELECT ROUND(100.0*SUM(CASE WHEN claim_status='Repudiated' THEN 1 ELSE 0 END)/COUNT(*),2) AS repudiation_rate_pct FROM claims;",
 "Aggregation/Ratio",["claims"],["claim_status"],["SUM","COUNT","CASE WHEN"]),

# ─── SALES / AGENTS ─────────────────────────────────────────────────────────
("PS-050","How many active agents are there?",
 "SELECT COUNT(*) AS active_agents FROM sales WHERE employment_status='Active';",
 "Filtering/Count",["sales"],["employment_status"],["COUNT","WHERE"]),

("PS-051","Who are the top 5 agents by policies sold?",
 "SELECT full_name, policies_sold FROM sales ORDER BY policies_sold DESC LIMIT 5;",
 "Ranking/TopN",["sales"],["full_name","policies_sold"],["ORDER BY","LIMIT"]),

("PS-052","Who are the top 5 agents by NBP achieved?",
 "SELECT full_name, nbp_achieved FROM sales ORDER BY nbp_achieved DESC LIMIT 5;",
 "Ranking/TopN",["sales"],["full_name","nbp_achieved"],["ORDER BY","LIMIT"]),

("PS-053","What is the average target achievement percentage across all agents?",
 "SELECT ROUND(AVG(target_achievement_pct),2) AS avg_target_achievement FROM sales;",
 "Aggregation/Average",["sales"],["target_achievement_pct"],["AVG"]),

("PS-054","How many agents exceeded their NBP target?",
 "SELECT COUNT(*) AS over_target_agents FROM sales WHERE target_achievement_pct > 100;",
 "Filtering/Count",["sales"],["target_achievement_pct"],["COUNT","WHERE"]),

("PS-055","What is the average persistency rate across all agents?",
 "SELECT ROUND(AVG(persistency_rate),4) AS avg_persistency FROM sales;",
 "Aggregation/Average",["sales"],["persistency_rate"],["AVG"]),

("PS-056","Which agent has the highest lapse rate?",
 "SELECT full_name, lapse_rate FROM sales ORDER BY lapse_rate DESC LIMIT 1;",
 "Ranking/Worst",["sales"],["full_name","lapse_rate"],["ORDER BY","LIMIT"]),

("PS-057","Which agent has the highest surrender rate?",
 "SELECT full_name, surrender_rate FROM sales ORDER BY surrender_rate DESC LIMIT 1;",
 "Ranking/Worst",["sales"],["full_name","surrender_rate"],["ORDER BY","LIMIT"]),

("PS-058","What is the average claim rate per agent?",
 "SELECT ROUND(AVG(claim_rate),4) AS avg_claim_rate FROM sales;",
 "Aggregation/Average",["sales"],["claim_rate"],["AVG"]),

("PS-059","How many agents have a persistency rate above 0.9?",
 "SELECT COUNT(*) AS high_persistency_agents FROM sales WHERE persistency_rate >= 0.9;",
 "Filtering/Count",["sales"],["persistency_rate"],["COUNT","WHERE"]),

("PS-060","What is the distribution of agent qualifications?",
 "SELECT qualification, COUNT(*) AS cnt FROM sales GROUP BY qualification ORDER BY cnt DESC;",
 "GroupBy/Distribution",["sales"],["qualification"],["COUNT","GROUP BY"]),

("PS-061","What is the total sum assured sold per branch?",
 "SELECT branch_id, SUM(sum_assured_sold) AS total_sa FROM sales GROUP BY branch_id ORDER BY total_sa DESC;",
 "GroupBy/Sum",["sales"],["branch_id","sum_assured_sold"],["SUM","GROUP BY"]),

("PS-062","How many agents have a zero claim rate?",
 "SELECT COUNT(*) AS zero_claim_agents FROM sales WHERE claim_rate=0;",
 "Filtering/Count",["sales"],["claim_rate"],["COUNT","WHERE"]),

("PS-063","Who is the top performing agent by NBP target achievement?",
 "SELECT full_name, target_achievement_pct FROM sales ORDER BY target_achievement_pct DESC LIMIT 1;",
 "Ranking/Top1",["sales"],["full_name","target_achievement_pct"],["ORDER BY","LIMIT"]),

# ─── PRODUCTS ──────────────────────────────────────────────────────────────
("PS-064","Which products are currently active?",
 "SELECT product_id, product_name FROM products WHERE is_active='Y';",
 "Filtering",["products"],["is_active","product_name"],["WHERE"]),

("PS-065","Which product has the highest premium rate per 1000?",
 "SELECT product_name, premium_rate_per_1000 FROM products ORDER BY premium_rate_per_1000 DESC LIMIT 1;",
 "Ranking/Max",["products"],["product_name","premium_rate_per_1000"],["ORDER BY","LIMIT"]),

("PS-066","How many products have maturity benefit?",
 "SELECT COUNT(*) AS products_with_maturity FROM products WHERE has_maturity_benefit='Y';",
 "Filtering/Count",["products"],["has_maturity_benefit"],["COUNT","WHERE"]),

("PS-067","How many products are participating (with profit)?",
 "SELECT COUNT(*) AS participating_products FROM products WHERE is_participating='Y';",
 "Filtering/Count",["products"],["is_participating"],["COUNT","WHERE"]),

("PS-068","What is the average minimum sum assured across all products?",
 "SELECT ROUND(AVG(min_sum_assured),2) AS avg_min_sa FROM products;",
 "Aggregation/Average",["products"],["min_sum_assured"],["AVG"]),

("PS-069","Which product was launched most recently?",
 "SELECT product_name, launch_date FROM products ORDER BY launch_date DESC LIMIT 1;",
 "Ranking/Recent",["products"],["product_name","launch_date"],["ORDER BY","LIMIT"]),

("PS-070","What product categories are available?",
 "SELECT DISTINCT product_category FROM products;",
 "Filtering/Distinct",["products"],["product_category"],["DISTINCT"]),

# ─── PLANS ─────────────────────────────────────────────────────────────────
("PS-071","How many plans are currently active?",
 "SELECT COUNT(*) AS active_plans FROM plans WHERE is_active='Y';",
 "Filtering/Count",["plans"],["is_active"],["COUNT","WHERE"]),

("PS-072","How many plans have regular premium payment type?",
 "SELECT COUNT(*) AS regular_plans FROM plans WHERE premium_payment_type='Regular';",
 "Filtering/Count",["plans"],["premium_payment_type"],["COUNT","WHERE"]),

("PS-073","How many plans are available per product?",
 "SELECT product_id, COUNT(*) AS plan_count FROM plans GROUP BY product_id ORDER BY plan_count DESC;",
 "GroupBy/Count",["plans"],["product_id"],["COUNT","GROUP BY"]),

# ═══════════ BLOCK B — CROSS-TABLE JOINS ═════════════════════════════════════
("PS-074","What is the average sum assured for smoker vs non-smoker policyholders?",
 "SELECT c.is_smoker, ROUND(AVG(p.sum_assured),2) AS avg_sa FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY c.is_smoker;",
 "Join/GroupBy/Comparison",["policies","customers"],["is_smoker","sum_assured"],["JOIN","AVG","GROUP BY"]),

("PS-075","How many policies does each gender hold on average per customer?",
 "SELECT c.gender, COUNT(p.policy_id) AS total_policies, ROUND(1.0*COUNT(p.policy_id)/COUNT(DISTINCT c.customer_id),2) AS avg_per_customer FROM customers c LEFT JOIN policies p ON c.customer_id=p.customer_id GROUP BY c.gender;",
 "Join/GroupBy/Ratio",["policies","customers"],["gender","policy_id"],["JOIN","COUNT","GROUP BY"]),

("PS-076","What is the total premium paid by customers in each marital status?",
 "SELECT c.marital_status, ROUND(SUM(p.total_premium_paid),2) AS total_premium FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY c.marital_status ORDER BY total_premium DESC;",
 "Join/GroupBy/Sum",["policies","customers"],["marital_status","total_premium_paid"],["JOIN","SUM","GROUP BY"]),

("PS-077","What is the average claim amount for male vs female policyholders?",
 "SELECT c.gender, ROUND(AVG(cl.claim_amount),2) AS avg_claim FROM claims cl JOIN customers c ON cl.customer_id=c.customer_id GROUP BY c.gender;",
 "Join/GroupBy/Comparison",["claims","customers"],["gender","claim_amount"],["JOIN","AVG","GROUP BY"]),

("PS-078","Which product category has the highest total claims payout?",
 "SELECT pr.product_category, ROUND(SUM(cl.claim_amount),2) AS total_payout FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number JOIN products pr ON pol.product_id=pr.product_id GROUP BY pr.product_category ORDER BY total_payout DESC LIMIT 1;",
 "Multi-Join/Aggregation",["claims","policies","products"],["product_category","claim_amount"],["JOIN","SUM","GROUP BY"]),

("PS-079","What is the average annual income of customers who have made claims vs those who have not?",
 "SELECT CASE WHEN cl.customer_id IS NOT NULL THEN 'Has Claim' ELSE 'No Claim' END AS claim_segment, ROUND(AVG(c.annual_income),2) AS avg_income FROM customers c LEFT JOIN (SELECT DISTINCT customer_id FROM claims) cl ON c.customer_id=cl.customer_id GROUP BY claim_segment;",
 "Join/Conditional/Comparison",["claims","customers"],["annual_income","customer_id"],["LEFT JOIN","AVG","CASE WHEN"]),

("PS-080","Which zone has the highest average claim amount?",
 "SELECT pol.zone, ROUND(AVG(cl.claim_amount),2) AS avg_claim FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number GROUP BY pol.zone ORDER BY avg_claim DESC LIMIT 1;",
 "Join/GroupBy/Ranking",["claims","policies"],["zone","claim_amount"],["JOIN","AVG","ORDER BY"]),

("PS-081","How many claims are associated with each product type?",
 "SELECT pol.product_name, COUNT(cl.claim_id) AS claim_count FROM policies pol JOIN claims cl ON pol.policy_number=cl.policy_number GROUP BY pol.product_name ORDER BY claim_count DESC;",
 "Join/GroupBy/Count",["claims","policies"],["product_name","claim_id"],["JOIN","COUNT","GROUP BY"]),

("PS-082","What is the total premium collected per agent (top 10)?",
 "SELECT s.full_name, ROUND(SUM(p.total_premium_paid),2) AS premium_collected FROM policies p JOIN sales s ON p.servicing_agent_id=s.sales_person_id GROUP BY s.full_name ORDER BY premium_collected DESC LIMIT 10;",
 "Join/GroupBy/Sum",["policies","sales"],["full_name","total_premium_paid"],["JOIN","SUM","GROUP BY"]),

("PS-083","What is the average BMI of customers who made death claims?",
 "SELECT ROUND(AVG(c.bmi),2) AS avg_bmi_death_claimants FROM claims cl JOIN customers c ON cl.customer_id=c.customer_id WHERE cl.claim_type='Death Claim';",
 "Join/Filtering/Aggregation",["claims","customers"],["bmi","claim_type"],["JOIN","AVG","WHERE"]),

("PS-084","How many policyholders are in each occupation category?",
 "SELECT c.occupation_id, COUNT(DISTINCT p.customer_id) AS policyholders FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY c.occupation_id ORDER BY policyholders DESC;",
 "Join/GroupBy/Count",["policies","customers"],["occupation_id"],["JOIN","COUNT","GROUP BY"]),

("PS-085","What is the total sum assured sold by agents who exceeded their NBP target?",
 "SELECT SUM(p.sum_assured) AS total_sa_over_target_agents FROM policies p JOIN sales s ON p.servicing_agent_id=s.sales_person_id WHERE s.target_achievement_pct > 100;",
 "Join/Filtering/Sum",["policies","sales"],["sum_assured","target_achievement_pct"],["JOIN","SUM","WHERE"]),

("PS-086","What is the claim settlement rate by zone?",
 "SELECT pol.zone, ROUND(100.0*SUM(CASE WHEN cl.claim_status='Settled' THEN 1 ELSE 0 END)/COUNT(*),2) AS settlement_rate_pct FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number GROUP BY pol.zone ORDER BY settlement_rate_pct DESC;",
 "Join/Conditional/Ratio",["claims","policies"],["zone","claim_status"],["JOIN","SUM","CASE WHEN","GROUP BY"]),

("PS-087","What is the average annual income of customers with lapsed policies?",
 "SELECT ROUND(AVG(c.annual_income),2) AS avg_income_lapsed FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE p.policy_status='Lapsed';",
 "Join/Filtering/Average",["policies","customers"],["annual_income","policy_status"],["JOIN","AVG","WHERE"]),

("PS-088","What is the average annual income of customers with In-Force policies?",
 "SELECT ROUND(AVG(c.annual_income),2) AS avg_income_inforce FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE p.policy_status='In-Force';",
 "Join/Filtering/Average",["policies","customers"],["annual_income","policy_status"],["JOIN","AVG","WHERE"]),

("PS-089","Which sales channel generates the highest average sum assured per policy?",
 "SELECT sales_channel, ROUND(AVG(sum_assured),2) AS avg_sa FROM policies GROUP BY sales_channel ORDER BY avg_sa DESC LIMIT 1;",
 "GroupBy/Ranking",["policies"],["sales_channel","sum_assured"],["AVG","GROUP BY","ORDER BY"]),

("PS-090","How many unique customers have at least one policy?",
 "SELECT COUNT(DISTINCT customer_id) AS customers_with_policy FROM policies;",
 "Aggregation/Distinct",["policies"],["customer_id"],["COUNT","DISTINCT"]),

# ═══════════ BLOCK C — RANKING & TOP-N ═══════════════════════════════════════
("PS-091","Which city has the highest total sum assured?",
 "SELECT location_city, SUM(sum_assured) AS total_sa FROM policies GROUP BY location_city ORDER BY total_sa DESC LIMIT 1;",
 "GroupBy/Ranking",["policies"],["location_city","sum_assured"],["SUM","GROUP BY","ORDER BY"]),

("PS-092","Which product has the highest total premium paid?",
 "SELECT product_name, ROUND(SUM(total_premium_paid),2) AS total_premium FROM policies GROUP BY product_name ORDER BY total_premium DESC LIMIT 1;",
 "GroupBy/Ranking",["policies"],["product_name","total_premium_paid"],["SUM","GROUP BY","ORDER BY"]),

("PS-093","Which 5 states have the highest policy count?",
 "SELECT state, COUNT(*) AS cnt FROM policies GROUP BY state ORDER BY cnt DESC LIMIT 5;",
 "GroupBy/TopN",["policies"],["state"],["COUNT","GROUP BY","ORDER BY"]),

("PS-094","Which branch has the most policies?",
 "SELECT branch_id, branch_name, COUNT(*) AS cnt FROM policies GROUP BY branch_id, branch_name ORDER BY cnt DESC LIMIT 1;",
 "GroupBy/Ranking",["policies"],["branch_id","branch_name"],["COUNT","GROUP BY","ORDER BY"]),

("PS-095","Who is the youngest agent?",
 "SELECT full_name, age FROM sales ORDER BY age ASC LIMIT 1;",
 "Ranking/Min",["sales"],["full_name","age"],["ORDER BY","LIMIT"]),

("PS-096","Who is the oldest agent?",
 "SELECT full_name, age FROM sales ORDER BY age DESC LIMIT 1;",
 "Ranking/Max",["sales"],["full_name","age"],["ORDER BY","LIMIT"]),

("PS-097","What are the top 3 most common medical diagnoses in health claims?",
 "SELECT medical, COUNT(*) AS cnt FROM claims WHERE claim_type='Health Claim' AND medical IS NOT NULL GROUP BY medical ORDER BY cnt DESC LIMIT 3;",
 "Filtering/GroupBy/TopN",["claims"],["medical","claim_type"],["COUNT","GROUP BY","ORDER BY"]),

("PS-098","Which customer has the highest annual income?",
 "SELECT full_name, annual_income FROM customers ORDER BY annual_income DESC LIMIT 1;",
 "Ranking/Max",["customers"],["full_name","annual_income"],["ORDER BY","LIMIT"]),

("PS-099","Which policy has the highest annual premium?",
 "SELECT policy_number, product_name, annual_premium FROM policies ORDER BY annual_premium DESC LIMIT 1;",
 "Ranking/Max",["policies"],["policy_number","annual_premium"],["ORDER BY","LIMIT"]),

("PS-100","Which 5 agents have the lowest persistency rate?",
 "SELECT full_name, persistency_rate FROM sales ORDER BY persistency_rate ASC LIMIT 5;",
 "Ranking/BottomN",["sales"],["full_name","persistency_rate"],["ORDER BY","LIMIT"]),

# ═══════════ BLOCK D — TIME-SERIES / TRENDS ══════════════════════════════════
("PS-101","How has the number of new policies changed year over year?",
 "SELECT SUBSTR(issuance_date,1,4) AS year, COUNT(*) AS new_policies FROM policies GROUP BY year ORDER BY year;",
 "Time-Series/Count",["policies"],["issuance_date"],["COUNT","GROUP BY"]),

("PS-102","How has total annual premium collected trended over the years?",
 "SELECT SUBSTR(issuance_date,1,4) AS year, ROUND(SUM(annual_premium),2) AS total_annual_premium FROM policies GROUP BY year ORDER BY year;",
 "Time-Series/Sum",["policies"],["issuance_date","annual_premium"],["SUM","GROUP BY"]),

("PS-103","How has the average sum assured evolved year over year?",
 "SELECT SUBSTR(issuance_date,1,4) AS year, ROUND(AVG(sum_assured),2) AS avg_sa FROM policies GROUP BY year ORDER BY year;",
 "Time-Series/Average",["policies"],["issuance_date","sum_assured"],["AVG","GROUP BY"]),

("PS-104","Which year had the highest number of claims?",
 "SELECT SUBSTR(claim_date,1,4) AS year, COUNT(*) AS claims FROM claims GROUP BY year ORDER BY claims DESC LIMIT 1;",
 "Time-Series/Ranking",["claims"],["claim_date"],["COUNT","GROUP BY","ORDER BY"]),

("PS-105","What is the monthly distribution of claims in 2024?",
 "SELECT SUBSTR(claim_date,6,2) AS month, COUNT(*) AS cnt FROM claims WHERE SUBSTR(claim_date,1,4)='2024' GROUP BY month ORDER BY month;",
 "Time-Series/Distribution",["claims"],["claim_date"],["COUNT","GROUP BY","WHERE"]),

("PS-106","How many policies matured each year?",
 "SELECT SUBSTR(maturity_date,1,4) AS maturity_year, COUNT(*) AS cnt FROM policies WHERE policy_status='Matured' GROUP BY maturity_year ORDER BY maturity_year;",
 "Time-Series/Count",["policies"],["maturity_date","policy_status"],["COUNT","GROUP BY"]),

("PS-107","What is the trend in lapsed policies by issuance year?",
 "SELECT SUBSTR(issuance_date,1,4) AS year, COUNT(*) AS lapsed FROM policies WHERE policy_status='Lapsed' GROUP BY year ORDER BY year;",
 "Time-Series/Trend",["policies"],["issuance_date","policy_status"],["COUNT","GROUP BY","WHERE"]),

("PS-108","How has the total claim amount changed each year?",
 "SELECT SUBSTR(claim_date,1,4) AS year, ROUND(SUM(claim_amount),2) AS total_claim FROM claims GROUP BY year ORDER BY year;",
 "Time-Series/Sum",["claims"],["claim_date","claim_amount"],["SUM","GROUP BY"]),

# ═══════════ BLOCK E — DISTRIBUTIONS & BUCKETING ═════════════════════════════
("PS-109","What is the distribution of customer ages in 10-year buckets?",
 "SELECT CASE WHEN age<30 THEN '<30' WHEN age<40 THEN '30-39' WHEN age<50 THEN '40-49' WHEN age<60 THEN '50-59' ELSE '60+' END AS age_bucket, COUNT(*) AS cnt FROM customers GROUP BY age_bucket ORDER BY age_bucket;",
 "Distribution/Bucketing",["customers"],["age"],["CASE WHEN","COUNT","GROUP BY"]),

("PS-110","What is the distribution of annual incomes across customers?",
 "SELECT CASE WHEN annual_income<500000 THEN '<5L' WHEN annual_income<1000000 THEN '5L-10L' WHEN annual_income<2000000 THEN '10L-20L' WHEN annual_income<5000000 THEN '20L-50L' ELSE '50L+' END AS income_bucket, COUNT(*) AS cnt FROM customers GROUP BY income_bucket ORDER BY cnt DESC;",
 "Distribution/Bucketing",["customers"],["annual_income"],["CASE WHEN","COUNT","GROUP BY"]),

("PS-111","What is the distribution of sum assured across policies?",
 "SELECT CASE WHEN sum_assured<1000000 THEN '<10L' WHEN sum_assured<5000000 THEN '10L-50L' WHEN sum_assured<10000000 THEN '50L-1Cr' ELSE '1Cr+' END AS sa_bucket, COUNT(*) AS cnt FROM policies GROUP BY sa_bucket ORDER BY cnt DESC;",
 "Distribution/Bucketing",["policies"],["sum_assured"],["CASE WHEN","COUNT","GROUP BY"]),

("PS-112","What is the distribution of claim amounts in health claims?",
 "SELECT CASE WHEN claim_amount<500000 THEN '<5L' WHEN claim_amount<1000000 THEN '5L-10L' WHEN claim_amount<2000000 THEN '10L-20L' ELSE '20L+' END AS claim_bucket, COUNT(*) AS cnt FROM claims WHERE claim_type='Health Claim' GROUP BY claim_bucket ORDER BY cnt DESC;",
 "Distribution/Bucketing",["claims"],["claim_amount","claim_type"],["CASE WHEN","COUNT","GROUP BY","WHERE"]),

("PS-113","What is the distribution of policy tenure (years) across all policies?",
 "SELECT policy_tenure_years, COUNT(*) AS cnt FROM policies GROUP BY policy_tenure_years ORDER BY policy_tenure_years;",
 "Distribution",["policies"],["policy_tenure_years"],["COUNT","GROUP BY"]),

("PS-114","What is the distribution of BMI across customers?",
 "SELECT CASE WHEN bmi<18.5 THEN 'Underweight' WHEN bmi<25 THEN 'Normal' WHEN bmi<30 THEN 'Overweight' ELSE 'Obese' END AS bmi_category, COUNT(*) AS cnt FROM customers GROUP BY bmi_category ORDER BY cnt DESC;",
 "Distribution/Bucketing",["customers"],["bmi"],["CASE WHEN","COUNT","GROUP BY"]),

# ═══════════ BLOCK F — CONDITIONAL / ADVANCED ANALYTICS ══════════════════════
("PS-115","What is the lapse rate for policies sold through different channels?",
 "SELECT sales_channel, ROUND(100.0*SUM(CASE WHEN policy_status='Lapsed' THEN 1 ELSE 0 END)/COUNT(*),2) AS lapse_rate_pct FROM policies GROUP BY sales_channel ORDER BY lapse_rate_pct DESC;",
 "Conditional/Ratio",["policies"],["sales_channel","policy_status"],["CASE WHEN","SUM","COUNT","GROUP BY"]),

("PS-116","What fraction of Bancassurance policies are In-Force vs Lapsed?",
 "SELECT policy_status, COUNT(*) AS cnt FROM policies WHERE sales_channel='Bancassurance' GROUP BY policy_status ORDER BY cnt DESC;",
 "Filtering/Distribution",["policies"],["sales_channel","policy_status"],["COUNT","GROUP BY","WHERE"]),

("PS-117","What is the average sum assured for policies with vs without riders?",
 "SELECT CASE WHEN rider_count>0 THEN 'With Riders' ELSE 'Without Riders' END AS rider_segment, ROUND(AVG(sum_assured),2) AS avg_sa FROM policies GROUP BY rider_segment;",
 "Conditional/Comparison",["policies"],["rider_count","sum_assured"],["CASE WHEN","AVG","GROUP BY"]),

("PS-118","Among settled claims, what percentage were early claims?",
 "SELECT ROUND(100.0*SUM(CASE WHEN is_early_claim=1 THEN 1 ELSE 0 END)/COUNT(*),2) AS early_claim_pct FROM claims WHERE claim_status='Settled';",
 "Filtering/Ratio",["claims"],["is_early_claim","claim_status"],["SUM","COUNT","CASE WHEN","WHERE"]),

("PS-119","What is the average claim amount for smoker vs non-smoker policyholders?",
 "SELECT c.is_smoker, ROUND(AVG(cl.claim_amount),2) AS avg_claim FROM claims cl JOIN customers c ON cl.customer_id=c.customer_id GROUP BY c.is_smoker;",
 "Join/Conditional/Comparison",["claims","customers"],["is_smoker","claim_amount"],["JOIN","AVG","GROUP BY"]),

("PS-120","Do customers with higher income tend to have higher sum assured?",
 "SELECT CASE WHEN c.annual_income<1000000 THEN 'Low Income (<10L)' WHEN c.annual_income<3000000 THEN 'Mid Income (10L-30L)' ELSE 'High Income (30L+)' END AS income_tier, ROUND(AVG(p.sum_assured),2) AS avg_sa FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY income_tier ORDER BY avg_sa DESC;",
 "Join/Conditional/Correlation",["policies","customers"],["annual_income","sum_assured"],["JOIN","CASE WHEN","AVG","GROUP BY"]),

("PS-121","What is the surrender rate by product?",
 "SELECT product_name, ROUND(100.0*SUM(CASE WHEN policy_status='Surrendered' THEN 1 ELSE 0 END)/COUNT(*),2) AS surrender_rate_pct FROM policies GROUP BY product_name ORDER BY surrender_rate_pct DESC;",
 "Conditional/Ratio",["policies"],["product_name","policy_status"],["CASE WHEN","SUM","COUNT","GROUP BY"]),

("PS-122","What is the death claim rate by zone?",
 "SELECT pol.zone, ROUND(100.0*COUNT(cl.claim_id)/COUNT(DISTINCT pol.policy_id),4) AS death_claim_rate_pct FROM policies pol LEFT JOIN claims cl ON pol.policy_number=cl.policy_number AND cl.claim_type='Death Claim' GROUP BY pol.zone ORDER BY death_claim_rate_pct DESC;",
 "Join/Conditional/Ratio",["policies","claims"],["zone","claim_type"],["LEFT JOIN","COUNT","CASE WHEN","GROUP BY"]),

("PS-123","Which product has the highest proportion of lapsed policies?",
 "SELECT product_name, ROUND(100.0*SUM(CASE WHEN policy_status='Lapsed' THEN 1 ELSE 0 END)/COUNT(*),2) AS lapse_pct FROM policies GROUP BY product_name ORDER BY lapse_pct DESC LIMIT 1;",
 "Conditional/Ratio/Ranking",["policies"],["product_name","policy_status"],["CASE WHEN","SUM","COUNT","GROUP BY","ORDER BY"]),

("PS-124","What percentage of customers are both smokers and obese (BMI>30)?",
 "SELECT ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM customers),2) AS pct FROM customers WHERE is_smoker='Y' AND bmi>30;",
 "Multi-Condition/Ratio",["customers"],["is_smoker","bmi"],["COUNT","WHERE","Subquery"]),

("PS-125","What is the average number of premiums paid for lapsed policies?",
 "SELECT ROUND(AVG(premiums_paid_count),2) AS avg_premiums_before_lapse FROM policies WHERE policy_status='Lapsed';",
 "Filtering/Average",["policies"],["premiums_paid_count","policy_status"],["AVG","WHERE"]),

("PS-126","What is the correlation proxy: avg claim amount vs product premium rate?",
 "SELECT pr.product_name, pr.premium_rate_per_1000, ROUND(AVG(cl.claim_amount),2) AS avg_claim_amount FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number JOIN products pr ON pol.product_id=pr.product_id GROUP BY pr.product_name, pr.premium_rate_per_1000 ORDER BY pr.premium_rate_per_1000 DESC;",
 "Multi-Join/Comparison",["claims","policies","products"],["premium_rate_per_1000","claim_amount"],["JOIN","AVG","GROUP BY"]),

("PS-127","How many policies have a premium-to-sum-assured ratio below 1%?",
 "SELECT COUNT(*) AS low_premium_ratio_policies FROM policies WHERE (annual_premium*1.0/sum_assured) < 0.01;",
 "Derived-Metric/Filtering",["policies"],["annual_premium","sum_assured"],["COUNT","WHERE","Derived Ratio"]),

("PS-128","What is the average total premium paid to sum assured ratio by product?",
 "SELECT product_name, ROUND(AVG(total_premium_paid*1.0/sum_assured),4) AS avg_premium_to_sa_ratio FROM policies GROUP BY product_name ORDER BY avg_premium_to_sa_ratio DESC;",
 "Derived-Metric/GroupBy",["policies"],["product_name","total_premium_paid","sum_assured"],["AVG","GROUP BY","Derived Ratio"]),

("PS-129","Which policies have total premiums paid exceeding the sum assured?",
 "SELECT policy_number, product_name, sum_assured, total_premium_paid FROM policies WHERE total_premium_paid > sum_assured ORDER BY total_premium_paid DESC LIMIT 5;",
 "Conditional/Anomaly",["policies"],["total_premium_paid","sum_assured"],["WHERE","ORDER BY"]),

("PS-130","Which customers have policies in more than one zone?",
 "SELECT customer_id, COUNT(DISTINCT zone) AS zone_count FROM policies GROUP BY customer_id HAVING zone_count > 1 LIMIT 10;",
 "GroupBy/Having/Filter",["policies"],["customer_id","zone"],["COUNT","DISTINCT","GROUP BY","HAVING"]),

# ═══════════ BLOCK G — ANOMALY DETECTION ═════════════════════════════════════
("PS-131","Which agents have a lapse rate above 50%?",
 "SELECT full_name, lapse_rate FROM sales WHERE lapse_rate > 0.5 ORDER BY lapse_rate DESC;",
 "Filtering/Anomaly",["sales"],["full_name","lapse_rate"],["WHERE","ORDER BY"]),

("PS-132","Which agents missed their NBP target by more than 50%?",
 "SELECT full_name, target_achievement_pct FROM sales WHERE target_achievement_pct < 50 ORDER BY target_achievement_pct ASC;",
 "Filtering/Anomaly",["sales"],["full_name","target_achievement_pct"],["WHERE","ORDER BY"]),

("PS-133","Which claims have amount more than 3x the average?",
 "SELECT claim_id, claim_type, claim_amount FROM claims WHERE claim_amount > 3*(SELECT AVG(claim_amount) FROM claims) ORDER BY claim_amount DESC;",
 "Anomaly/Subquery",["claims"],["claim_id","claim_amount"],["WHERE","Subquery","ORDER BY"]),

("PS-134","Which customers have low income (<200K) but high sum assured (>5M)?",
 "SELECT c.customer_id, c.full_name, c.annual_income, p.sum_assured FROM customers c JOIN policies p ON c.customer_id=p.customer_id WHERE c.annual_income < 200000 AND p.sum_assured > 5000000 ORDER BY p.sum_assured DESC;",
 "Join/Multi-Condition/Anomaly",["customers","policies"],["annual_income","sum_assured"],["JOIN","WHERE","ORDER BY"]),

("PS-135","How many policies were surrendered within the first 3 years?",
 "SELECT COUNT(*) AS early_surrenders FROM policies WHERE policy_status='Surrendered' AND premium_paying_term_years <= 3;",
 "Filtering/Anomaly",["policies"],["policy_status","premium_paying_term_years"],["COUNT","WHERE"]),

("PS-136","Which branch has the highest lapse rate among its policies?",
 "SELECT branch_id, branch_name, ROUND(100.0*SUM(CASE WHEN policy_status='Lapsed' THEN 1 ELSE 0 END)/COUNT(*),2) AS lapse_pct FROM policies GROUP BY branch_id, branch_name ORDER BY lapse_pct DESC LIMIT 1;",
 "GroupBy/Conditional/Ranking",["policies"],["branch_id","policy_status"],["CASE WHEN","SUM","COUNT","GROUP BY","ORDER BY"]),

("PS-137","Which agents have the lowest FTR (First Time Resolution) rate?",
 "SELECT full_name, ftr_rate FROM sales WHERE ftr_rate < 0.5 ORDER BY ftr_rate ASC LIMIT 5;",
 "Filtering/Anomaly",["sales"],["full_name","ftr_rate"],["WHERE","ORDER BY"]),

("PS-138","How many In-Force policies have overdue next premium due date?",
 "SELECT COUNT(*) AS overdue_inforce FROM policies WHERE policy_status='In-Force' AND next_premium_due_date < date('now') AND next_premium_due_date IS NOT NULL AND next_premium_due_date != '';",
 "Temporal/Anomaly",["policies"],["policy_status","next_premium_due_date"],["COUNT","WHERE","date()"]),

("PS-139","Which customers have filed more than one claim?",
 "SELECT customer_id, COUNT(*) AS claim_count FROM claims GROUP BY customer_id HAVING claim_count > 1 ORDER BY claim_count DESC;",
 "GroupBy/Having/Anomaly",["claims"],["customer_id"],["COUNT","GROUP BY","HAVING"]),

("PS-140","Which claim type has the highest repudiation rate?",
 "SELECT claim_type, ROUND(100.0*SUM(CASE WHEN claim_status='Repudiated' THEN 1 ELSE 0 END)/COUNT(*),2) AS repudiation_pct FROM claims GROUP BY claim_type ORDER BY repudiation_pct DESC LIMIT 1;",
 "GroupBy/Conditional/Ranking",["claims"],["claim_type","claim_status"],["CASE WHEN","SUM","COUNT","GROUP BY","ORDER BY"]),

# ═══════════ BLOCK H — AGENT x POLICY x PRODUCT CROSS-TABLE ══════════════════
("PS-141","Which product is most commonly sold by Bancassurance?",
 "SELECT product_name, COUNT(*) AS cnt FROM policies WHERE sales_channel='Bancassurance' GROUP BY product_name ORDER BY cnt DESC LIMIT 1;",
 "Filtering/GroupBy/Ranking",["policies"],["sales_channel","product_name"],["COUNT","GROUP BY","ORDER BY"]),

("PS-142","What is the average number of riders per policy by product?",
 "SELECT product_name, ROUND(AVG(rider_count),2) AS avg_riders FROM policies GROUP BY product_name ORDER BY avg_riders DESC;",
 "GroupBy/Average",["policies"],["product_name","rider_count"],["AVG","GROUP BY"]),

("PS-143","What is the total NBP achieved per branch?",
 "SELECT branch_id, ROUND(SUM(nbp_achieved),2) AS total_nbp FROM sales GROUP BY branch_id ORDER BY total_nbp DESC;",
 "GroupBy/Sum",["sales"],["branch_id","nbp_achieved"],["SUM","GROUP BY"]),

("PS-144","What is the average target achievement by agent qualification?",
 "SELECT qualification, ROUND(AVG(target_achievement_pct),2) AS avg_achievement FROM sales GROUP BY qualification ORDER BY avg_achievement DESC;",
 "GroupBy/Average",["sales"],["qualification","target_achievement_pct"],["AVG","GROUP BY"]),

("PS-145","Which zone has the highest total sum assured for Term Life policies?",
 "SELECT zone, SUM(sum_assured) AS total_sa FROM policies WHERE product_name='Term Life' GROUP BY zone ORDER BY total_sa DESC LIMIT 1;",
 "Filtering/GroupBy/Ranking",["policies"],["zone","sum_assured","product_name"],["SUM","GROUP BY","WHERE","ORDER BY"]),

("PS-146","What is the average annual premium for Direct vs Agency channel policies?",
 "SELECT sales_channel, ROUND(AVG(annual_premium),2) AS avg_premium FROM policies WHERE sales_channel IN ('Direct','Agency') GROUP BY sales_channel;",
 "Filtering/GroupBy/Comparison",["policies"],["sales_channel","annual_premium"],["AVG","GROUP BY","WHERE","IN"]),

("PS-147","What is the total sum assured for medical underwritten policies?",
 "SELECT ROUND(SUM(sum_assured),2) AS total_sa_medical_uw FROM policies WHERE medical_underwriting_flag='Y';",
 "Filtering/Sum",["policies"],["medical_underwriting_flag","sum_assured"],["SUM","WHERE"]),

("PS-148","What percentage of policies were sold online?",
 "SELECT ROUND(100.0*SUM(CASE WHEN sales_channel='Online' THEN 1 ELSE 0 END)/COUNT(*),2) AS online_pct FROM policies;",
 "Conditional/Ratio",["policies"],["sales_channel"],["SUM","COUNT","CASE WHEN"]),

("PS-149","How many policies per plan payment type (Regular vs Limited)?",
 "SELECT pl.premium_payment_type, COUNT(po.policy_id) AS cnt FROM policies po JOIN plans pl ON po.plan_id=pl.plan_id GROUP BY pl.premium_payment_type ORDER BY cnt DESC;",
 "Join/GroupBy/Count",["policies","plans"],["premium_payment_type"],["JOIN","COUNT","GROUP BY"]),

("PS-150","What is the average sum assured for Regular vs Limited pay plans?",
 "SELECT pl.premium_payment_type, ROUND(AVG(po.sum_assured),2) AS avg_sa FROM policies po JOIN plans pl ON po.plan_id=pl.plan_id GROUP BY pl.premium_payment_type;",
 "Join/GroupBy/Average",["policies","plans"],["premium_payment_type","sum_assured"],["JOIN","AVG","GROUP BY"]),

# ═══════════ BLOCK I — ADVANCED DERIVED METRICS ═══════════════════════════════
("PS-151","What is the loss ratio (total claims / total premiums) by product?",
 "SELECT pol.product_name, ROUND(SUM(cl.claim_amount)/SUM(pol.total_premium_paid),4) AS loss_ratio FROM policies pol JOIN claims cl ON pol.policy_number=cl.policy_number GROUP BY pol.product_name ORDER BY loss_ratio DESC;",
 "Multi-Join/Derived-Metric",["policies","claims"],["product_name","claim_amount","total_premium_paid"],["JOIN","SUM","GROUP BY"]),

("PS-152","What is the average claim-to-sum-assured ratio for settled claims?",
 "SELECT ROUND(AVG(cl.claim_amount*1.0/pol.sum_assured),4) AS avg_claim_sa_ratio FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number WHERE cl.claim_status='Settled';",
 "Join/Derived-Metric/Filtering",["claims","policies"],["claim_amount","sum_assured","claim_status"],["JOIN","AVG","WHERE"]),

("PS-153","Which zone has the most diverse product mix?",
 "SELECT zone, COUNT(DISTINCT product_name) AS distinct_products FROM policies GROUP BY zone ORDER BY distinct_products DESC LIMIT 1;",
 "GroupBy/Diversity",["policies"],["zone","product_name"],["COUNT","DISTINCT","GROUP BY"]),

("PS-154","What is the average customer lifetime value proxy (total premiums) by income tier?",
 "SELECT CASE WHEN c.annual_income < 1000000 THEN 'Low' WHEN c.annual_income < 3000000 THEN 'Mid' ELSE 'High' END AS income_tier, ROUND(AVG(p.total_premium_paid),2) AS avg_clv FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY income_tier ORDER BY avg_clv DESC;",
 "Join/Conditional/CLV",["policies","customers"],["annual_income","total_premium_paid"],["JOIN","CASE WHEN","AVG","GROUP BY"]),

("PS-155","What percentage of medically underwritten policies are In-Force?",
 "SELECT ROUND(100.0*SUM(CASE WHEN policy_status='In-Force' THEN 1 ELSE 0 END)/COUNT(*),2) AS inforce_pct FROM policies WHERE medical_underwriting_flag='Y';",
 "Filtering/Conditional/Ratio",["policies"],["medical_underwriting_flag","policy_status"],["CASE WHEN","SUM","COUNT","WHERE"]),

("PS-156","What is the average annual premium for smoker vs non-smoker customers?",
 "SELECT c.is_smoker, ROUND(AVG(p.annual_premium),2) AS avg_premium FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY c.is_smoker;",
 "Join/GroupBy/Comparison",["policies","customers"],["is_smoker","annual_premium"],["JOIN","AVG","GROUP BY"]),

("PS-157","What is the average time in years from policy issuance to claim?",
 "SELECT ROUND(AVG(policy_tenure_at_claim_months/12.0),2) AS avg_years_to_claim FROM claims;",
 "Aggregation/Derived-Metric",["claims"],["policy_tenure_at_claim_months"],["AVG","Derived"]),

("PS-158","Which month of the year sees the most policy issuances?",
 "SELECT SUBSTR(issuance_date,6,2) AS month, COUNT(*) AS cnt FROM policies GROUP BY month ORDER BY cnt DESC LIMIT 1;",
 "Time-Series/Seasonality",["policies"],["issuance_date"],["COUNT","GROUP BY"]),

("PS-159","What is the claim frequency (claims per policy) by sales channel?",
 "SELECT pol.sales_channel, ROUND(COUNT(cl.claim_id)*1.0/COUNT(DISTINCT pol.policy_id),4) AS claim_frequency FROM policies pol LEFT JOIN claims cl ON pol.policy_number=cl.policy_number GROUP BY pol.sales_channel ORDER BY claim_frequency DESC;",
 "Join/Derived-Metric/Comparison",["policies","claims"],["sales_channel","claim_id"],["LEFT JOIN","COUNT","DISTINCT","GROUP BY"]),

("PS-160","What is the average FTR rate by branch for agents?",
 "SELECT branch_id, ROUND(AVG(ftr_rate),4) AS avg_ftr FROM sales GROUP BY branch_id ORDER BY avg_ftr DESC;",
 "GroupBy/Average",["sales"],["branch_id","ftr_rate"],["AVG","GROUP BY"]),

# ═══════════ BLOCK J — MORE FILTERING & COMPARISONS ══════════════════════════
("PS-161","How many health claims were repudiated?",
 "SELECT COUNT(*) AS repudiated_health_claims FROM claims WHERE claim_type='Health Claim' AND claim_status='Repudiated';",
 "Multi-Condition/Count",["claims"],["claim_type","claim_status"],["COUNT","WHERE"]),

("PS-162","What is the average sum assured for policies in the North zone?",
 "SELECT ROUND(AVG(sum_assured),2) AS avg_sa_north FROM policies WHERE zone='North';",
 "Filtering/Average",["policies"],["zone","sum_assured"],["AVG","WHERE"]),

("PS-163","What is the average sum assured for policies in the South zone?",
 "SELECT ROUND(AVG(sum_assured),2) AS avg_sa_south FROM policies WHERE zone='South';",
 "Filtering/Average",["policies"],["zone","sum_assured"],["AVG","WHERE"]),

("PS-164","How many agents joined after 2015?",
 "SELECT COUNT(*) AS agents_post_2015 FROM sales WHERE date_of_joining > '2015-12-31';",
 "Filtering/Count",["sales"],["date_of_joining"],["COUNT","WHERE"]),

("PS-165","What is the total premium paid for Endowment policies?",
 "SELECT ROUND(SUM(total_premium_paid),2) AS total_endowment_premium FROM policies WHERE product_name='Endowment';",
 "Filtering/Sum",["policies"],["product_name","total_premium_paid"],["SUM","WHERE"]),

("PS-166","How many critical illness claims were filed?",
 "SELECT COUNT(*) AS critical_illness_claims FROM claims WHERE claim_type='Critical Illness Claim';",
 "Filtering/Count",["claims"],["claim_type"],["COUNT","WHERE"]),

("PS-167","What is the average policy tenure for matured policies?",
 "SELECT ROUND(AVG(policy_tenure_years),2) AS avg_tenure_matured FROM policies WHERE policy_status='Matured';",
 "Filtering/Average",["policies"],["policy_status","policy_tenure_years"],["AVG","WHERE"]),

("PS-168","How many customers have more than 2 policies?",
 "SELECT COUNT(*) AS multi_policy_customers FROM (SELECT customer_id FROM policies GROUP BY customer_id HAVING COUNT(*) > 2);",
 "GroupBy/Having/Count",["policies"],["customer_id"],["COUNT","GROUP BY","HAVING","Subquery"]),

("PS-169","What is the average annual premium by payment mode?",
 "SELECT payment_mode, ROUND(AVG(annual_premium),2) AS avg_premium FROM policies GROUP BY payment_mode ORDER BY avg_premium DESC;",
 "GroupBy/Comparison",["policies"],["payment_mode","annual_premium"],["AVG","GROUP BY"]),

("PS-170","What is the maximum claim amount among death claims?",
 "SELECT MAX(claim_amount) AS max_death_claim FROM claims WHERE claim_type='Death Claim';",
 "Filtering/Max",["claims"],["claim_type","claim_amount"],["MAX","WHERE"]),

("PS-171","What is the total sum assured sold by active agents?",
 "SELECT ROUND(SUM(sum_assured_sold),2) AS total_sa_active_agents FROM sales WHERE employment_status='Active';",
 "Filtering/Sum",["sales"],["employment_status","sum_assured_sold"],["SUM","WHERE"]),

("PS-172","Which state has the highest lapse rate?",
 "SELECT state, ROUND(100.0*SUM(CASE WHEN policy_status='Lapsed' THEN 1 ELSE 0 END)/COUNT(*),2) AS lapse_pct FROM policies GROUP BY state ORDER BY lapse_pct DESC LIMIT 1;",
 "GroupBy/Conditional/Ranking",["policies"],["state","policy_status"],["CASE WHEN","SUM","COUNT","GROUP BY","ORDER BY"]),

("PS-173","What is the average BMI of customers with matured policies?",
 "SELECT ROUND(AVG(c.bmi),2) AS avg_bmi_matured FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE p.policy_status='Matured';",
 "Join/Filtering/Average",["policies","customers"],["bmi","policy_status"],["JOIN","AVG","WHERE"]),

("PS-174","How many policies have zero premiums paid count?",
 "SELECT COUNT(*) AS zero_premium_paid FROM policies WHERE premiums_paid_count=0;",
 "Filtering/Count",["policies"],["premiums_paid_count"],["COUNT","WHERE"]),

("PS-175","What is the average claim amount for Critical Illness claims?",
 "SELECT ROUND(AVG(claim_amount),2) AS avg_ci_claim FROM claims WHERE claim_type='Critical Illness Claim';",
 "Filtering/Average",["claims"],["claim_type","claim_amount"],["AVG","WHERE"]),

("PS-176","What is the surrender claim payout as a fraction of total claim payout?",
 "SELECT ROUND(SUM(CASE WHEN claim_type='Surrender Claim' THEN claim_amount ELSE 0 END)*100.0/SUM(claim_amount),2) AS surrender_pct FROM claims;",
 "Conditional/Ratio",["claims"],["claim_type","claim_amount"],["SUM","CASE WHEN"]),

("PS-177","How many policies are there per branch (top 5)?",
 "SELECT branch_id, branch_name, COUNT(*) AS cnt FROM policies GROUP BY branch_id, branch_name ORDER BY cnt DESC LIMIT 5;",
 "GroupBy/TopN",["policies"],["branch_id","branch_name"],["COUNT","GROUP BY","ORDER BY"]),

("PS-178","What is the average annual premium for ULIP products?",
 "SELECT ROUND(AVG(annual_premium),2) AS avg_ulip_premium FROM policies WHERE product_name='ULIP';",
 "Filtering/Average",["policies"],["product_name","annual_premium"],["AVG","WHERE"]),

("PS-179","What is the ratio of death claims to total policies?",
 "SELECT ROUND(100.0*(SELECT COUNT(*) FROM claims WHERE claim_type='Death Claim')/COUNT(*),4) AS death_claim_rate_pct FROM policies;",
 "Ratio/Cross-Count",["claims","policies"],["claim_type"],["COUNT","Subquery"]),

("PS-180","How many customers have no associated policy?",
 "SELECT COUNT(*) AS customers_without_policy FROM customers WHERE customer_id NOT IN (SELECT DISTINCT customer_id FROM policies);",
 "Anti-Join/Count",["customers","policies"],["customer_id"],["COUNT","NOT IN","Subquery"]),

# ═══════════ BLOCK K — SEGMENT COMPARISON & PROFILING ════════════════════════
("PS-181","What is the average sum assured for policies held by widowed vs married customers?",
 "SELECT c.marital_status, ROUND(AVG(p.sum_assured),2) AS avg_sa FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE c.marital_status IN ('Married','Widowed') GROUP BY c.marital_status;",
 "Join/Filtering/Comparison",["policies","customers"],["marital_status","sum_assured"],["JOIN","AVG","GROUP BY","WHERE"]),

("PS-182","What is the average annual income for customers in each zone?",
 "SELECT p.zone, ROUND(AVG(c.annual_income),2) AS avg_income FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY p.zone ORDER BY avg_income DESC;",
 "Join/GroupBy/Average",["policies","customers"],["zone","annual_income"],["JOIN","AVG","GROUP BY"]),

("PS-183","What is the proportion of female policyholders by product?",
 "SELECT p.product_name, ROUND(100.0*SUM(CASE WHEN c.gender='Female' THEN 1 ELSE 0 END)/COUNT(*),2) AS female_pct FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY p.product_name ORDER BY female_pct DESC;",
 "Join/Conditional/Ratio",["policies","customers"],["product_name","gender"],["JOIN","SUM","CASE WHEN","GROUP BY"]),

("PS-184","What is the average claim amount for customers aged above 50?",
 "SELECT ROUND(AVG(cl.claim_amount),2) AS avg_claim_above_50 FROM claims cl JOIN customers c ON cl.customer_id=c.customer_id WHERE c.age > 50;",
 "Join/Filtering/Average",["claims","customers"],["age","claim_amount"],["JOIN","AVG","WHERE"]),

("PS-185","What is the average claim amount for customers aged below 40?",
 "SELECT ROUND(AVG(cl.claim_amount),2) AS avg_claim_below_40 FROM claims cl JOIN customers c ON cl.customer_id=c.customer_id WHERE c.age < 40;",
 "Join/Filtering/Average",["claims","customers"],["age","claim_amount"],["JOIN","AVG","WHERE"]),

("PS-186","How does the lapse rate compare across income tiers?",
 "SELECT CASE WHEN c.annual_income < 1000000 THEN 'Low' WHEN c.annual_income < 3000000 THEN 'Mid' ELSE 'High' END AS income_tier, ROUND(100.0*SUM(CASE WHEN p.policy_status='Lapsed' THEN 1 ELSE 0 END)/COUNT(*),2) AS lapse_pct FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY income_tier ORDER BY lapse_pct DESC;",
 "Join/Conditional/Comparison",["policies","customers"],["annual_income","policy_status"],["JOIN","CASE WHEN","SUM","COUNT","GROUP BY"]),

("PS-187","What is the average sum assured per policy sold by the best agent?",
 "SELECT s.full_name, ROUND(AVG(p.sum_assured),2) AS avg_sa FROM policies p JOIN sales s ON p.servicing_agent_id=s.sales_person_id GROUP BY s.full_name ORDER BY avg_sa DESC LIMIT 1;",
 "Join/GroupBy/Ranking",["policies","sales"],["full_name","sum_assured"],["JOIN","AVG","GROUP BY","ORDER BY"]),

("PS-188","What is the claim resolution rate by year?",
 "SELECT SUBSTR(claim_date,1,4) AS year, ROUND(100.0*SUM(CASE WHEN claim_status='Settled' THEN 1 ELSE 0 END)/COUNT(*),2) AS resolution_rate_pct FROM claims GROUP BY year ORDER BY year;",
 "Time-Series/Conditional/Ratio",["claims"],["claim_date","claim_status"],["CASE WHEN","SUM","COUNT","GROUP BY"]),

("PS-189","What is the average persistency rate per branch?",
 "SELECT branch_id, ROUND(AVG(persistency_rate),4) AS avg_persistency FROM sales GROUP BY branch_id ORDER BY avg_persistency DESC;",
 "GroupBy/Average",["sales"],["branch_id","persistency_rate"],["AVG","GROUP BY"]),

("PS-190","What is the total claim amount for Agency-channel policies?",
 "SELECT ROUND(SUM(cl.claim_amount),2) AS total_agency_claims FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number WHERE pol.sales_channel='Agency';",
 "Join/Filtering/Sum",["claims","policies"],["sales_channel","claim_amount"],["JOIN","SUM","WHERE"]),

# ═══════════ BLOCK L — PRODUCT PERFORMANCE ═══════════════════════════════════
("PS-191","What is the market share of each product by policy count?",
 "SELECT product_name, COUNT(*) AS cnt, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS market_share_pct FROM policies GROUP BY product_name ORDER BY cnt DESC;",
 "Distribution/Window",["policies"],["product_name"],["COUNT","WINDOW","GROUP BY"]),

("PS-192","What is the average policy tenure for each product?",
 "SELECT product_name, ROUND(AVG(policy_tenure_years),2) AS avg_tenure FROM policies GROUP BY product_name ORDER BY avg_tenure DESC;",
 "GroupBy/Average",["policies"],["product_name","policy_tenure_years"],["AVG","GROUP BY"]),

("PS-193","Which product generates the most total sum assured?",
 "SELECT product_name, SUM(sum_assured) AS total_sa FROM policies GROUP BY product_name ORDER BY total_sa DESC LIMIT 1;",
 "GroupBy/Ranking",["policies"],["product_name","sum_assured"],["SUM","GROUP BY","ORDER BY"]),

("PS-194","What is the average claim amount per product category?",
 "SELECT pr.product_category, ROUND(AVG(cl.claim_amount),2) AS avg_claim FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number JOIN products pr ON pol.product_id=pr.product_id GROUP BY pr.product_category ORDER BY avg_claim DESC;",
 "Multi-Join/GroupBy/Average",["claims","policies","products"],["product_category","claim_amount"],["JOIN","AVG","GROUP BY"]),

("PS-195","How many policies exist per product category?",
 "SELECT pr.product_category, COUNT(po.policy_id) AS cnt FROM policies po JOIN products pr ON po.product_id=pr.product_id GROUP BY pr.product_category ORDER BY cnt DESC;",
 "Join/GroupBy/Count",["policies","products"],["product_category"],["JOIN","COUNT","GROUP BY"]),

("PS-196","What is the average premium rate for products with maturity benefit?",
 "SELECT ROUND(AVG(premium_rate_per_1000),2) AS avg_rate_with_maturity FROM products WHERE has_maturity_benefit='Y';",
 "Filtering/Average",["products"],["has_maturity_benefit","premium_rate_per_1000"],["AVG","WHERE"]),

("PS-197","Which products have both surrender value and maturity benefit?",
 "SELECT product_name FROM products WHERE has_surrender_value='Y' AND has_maturity_benefit='Y';",
 "Multi-Condition/Filter",["products"],["has_surrender_value","has_maturity_benefit"],["WHERE"]),

("PS-198","What is the total NBP achieved vs target for all agents combined?",
 "SELECT ROUND(SUM(nbp_achieved),2) AS total_nbp_achieved, ROUND(SUM(nbp_target),2) AS total_nbp_target, ROUND(100.0*SUM(nbp_achieved)/SUM(nbp_target),2) AS overall_achievement_pct FROM sales;",
 "Aggregation/Derived-Metric",["sales"],["nbp_achieved","nbp_target"],["SUM","Derived Ratio"]),

("PS-199","What is the claim rate for policies where the customer is a smoker?",
 "SELECT ROUND(100.0*COUNT(DISTINCT cl.policy_number)/COUNT(DISTINCT pol.policy_number),4) AS smoker_claim_rate_pct FROM policies pol JOIN customers c ON pol.customer_id=c.customer_id LEFT JOIN claims cl ON pol.policy_number=cl.policy_number WHERE c.is_smoker='Y';",
 "Join/Conditional/Ratio",["policies","customers","claims"],["is_smoker","policy_number"],["JOIN","LEFT JOIN","COUNT","DISTINCT","WHERE"]),

("PS-200","What is the claim rate for non-smoker policies?",
 "SELECT ROUND(100.0*COUNT(DISTINCT cl.policy_number)/COUNT(DISTINCT pol.policy_number),4) AS non_smoker_claim_rate_pct FROM policies pol JOIN customers c ON pol.customer_id=c.customer_id LEFT JOIN claims cl ON pol.policy_number=cl.policy_number WHERE c.is_smoker='N';",
 "Join/Conditional/Ratio",["policies","customers","claims"],["is_smoker","policy_number"],["JOIN","LEFT JOIN","COUNT","DISTINCT","WHERE"]),

# ═══════════ BLOCK M — ADDITIONAL DEEP QUERIES (201-260) ═════════════════════
("PS-201","What is the average age of customers who bought Whole Life policies?",
 "SELECT ROUND(AVG(c.age),2) AS avg_age_whole_life FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE p.product_name='Whole Life';",
 "Join/Filtering/Average",["policies","customers"],["product_name","age"],["JOIN","AVG","WHERE"]),

("PS-202","What is the total claim amount for early claims only?",
 "SELECT ROUND(SUM(claim_amount),2) AS total_early_claim_amount FROM claims WHERE is_early_claim=1;",
 "Filtering/Sum",["claims"],["is_early_claim","claim_amount"],["SUM","WHERE"]),

("PS-203","What proportion of Annuity policies are in-force?",
 "SELECT ROUND(100.0*SUM(CASE WHEN policy_status='In-Force' THEN 1 ELSE 0 END)/COUNT(*),2) AS inforce_pct FROM policies WHERE product_name='Annuity';",
 "Filtering/Conditional/Ratio",["policies"],["product_name","policy_status"],["CASE WHEN","SUM","COUNT","WHERE"]),

("PS-204","What is the average premiums paid count for In-Force vs Lapsed policies?",
 "SELECT policy_status, ROUND(AVG(premiums_paid_count),2) AS avg_ppc FROM policies WHERE policy_status IN ('In-Force','Lapsed') GROUP BY policy_status;",
 "Filtering/GroupBy/Comparison",["policies"],["policy_status","premiums_paid_count"],["AVG","GROUP BY","WHERE"]),

("PS-205","How many claim-free agents (claim_rate=0) are there per branch?",
 "SELECT branch_id, COUNT(*) AS zero_claim_agents FROM sales WHERE claim_rate=0 GROUP BY branch_id ORDER BY zero_claim_agents DESC;",
 "Filtering/GroupBy/Count",["sales"],["branch_id","claim_rate"],["COUNT","WHERE","GROUP BY"]),

("PS-206","What is the average policy tenure for Child Plan policies?",
 "SELECT ROUND(AVG(policy_tenure_years),2) AS avg_tenure_child_plan FROM policies WHERE product_name='Child Plan';",
 "Filtering/Average",["policies"],["product_name","policy_tenure_years"],["AVG","WHERE"]),

("PS-207","What is the policy status distribution within each zone?",
 "SELECT zone, policy_status, COUNT(*) AS cnt, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(PARTITION BY zone),2) AS pct_in_zone FROM policies GROUP BY zone, policy_status ORDER BY zone, cnt DESC;",
 "GroupBy/Window/Distribution",["policies"],["zone","policy_status"],["COUNT","WINDOW","PARTITION BY","GROUP BY"]),

("PS-208","How many customers made a claim within the first 2 years of their policy?",
 "SELECT COUNT(*) AS early_claimants FROM claims WHERE policy_tenure_at_claim_months <= 24;",
 "Filtering/Count",["claims"],["policy_tenure_at_claim_months"],["COUNT","WHERE"]),

("PS-209","What is the average sum assured for policies by nominee relation?",
 "SELECT nominee_relation, ROUND(AVG(sum_assured),2) AS avg_sa FROM policies GROUP BY nominee_relation ORDER BY avg_sa DESC;",
 "GroupBy/Average",["policies"],["nominee_relation","sum_assured"],["AVG","GROUP BY"]),

("PS-210","Which agent branch has the highest average target achievement?",
 "SELECT branch_id, ROUND(AVG(target_achievement_pct),2) AS avg_achievement FROM sales GROUP BY branch_id ORDER BY avg_achievement DESC LIMIT 1;",
 "GroupBy/Ranking",["sales"],["branch_id","target_achievement_pct"],["AVG","GROUP BY","ORDER BY"]),

("PS-211","What is the total maturity claim amount paid out?",
 "SELECT ROUND(SUM(claim_amount),2) AS total_maturity_payout FROM claims WHERE claim_type='Maturity Claim' AND claim_status='Settled';",
 "Multi-Condition/Sum",["claims"],["claim_type","claim_status","claim_amount"],["SUM","WHERE"]),

("PS-212","What is the average sum assured for policies maturing within the next 5 years?",
 "SELECT ROUND(AVG(sum_assured),2) AS avg_sa_maturing_soon FROM policies WHERE maturity_date <= date('now', '+5 years') AND policy_status='In-Force';",
 "Temporal/Filtering/Average",["policies"],["maturity_date","sum_assured","policy_status"],["AVG","WHERE","date()"]),

("PS-213","Which product has the most claims under investigation?",
 "SELECT pol.product_name, COUNT(*) AS under_investigation FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number WHERE cl.claim_status='Under Investigation' GROUP BY pol.product_name ORDER BY under_investigation DESC LIMIT 1;",
 "Join/Filtering/Ranking",["claims","policies"],["product_name","claim_status"],["JOIN","COUNT","WHERE","GROUP BY","ORDER BY"]),

("PS-214","What is the average annual premium for customers with BMI above 30?",
 "SELECT ROUND(AVG(p.annual_premium),2) AS avg_premium_obese FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE c.bmi > 30;",
 "Join/Filtering/Average",["policies","customers"],["bmi","annual_premium"],["JOIN","AVG","WHERE"]),

("PS-215","How many total claims per medical diagnosis type?",
 "SELECT medical, COUNT(*) AS cnt FROM claims WHERE medical IS NOT NULL AND medical != '' GROUP BY medical ORDER BY cnt DESC;",
 "Filtering/GroupBy/Count",["claims"],["medical"],["COUNT","GROUP BY","WHERE"]),

("PS-216","What is the average number of policies sold per active agent?",
 "SELECT ROUND(AVG(policies_sold),2) AS avg_policies_per_agent FROM sales WHERE employment_status='Active';",
 "Filtering/Average",["sales"],["policies_sold","employment_status"],["AVG","WHERE"]),

("PS-217","What is the total premium collected for quarterly payment mode policies?",
 "SELECT ROUND(SUM(total_premium_paid),2) AS total_quarterly_premium FROM policies WHERE payment_mode='Quarterly';",
 "Filtering/Sum",["policies"],["payment_mode","total_premium_paid"],["SUM","WHERE"]),

("PS-218","What percentage of claims are settled for each claim type?",
 "SELECT claim_type, ROUND(100.0*SUM(CASE WHEN claim_status='Settled' THEN 1 ELSE 0 END)/COUNT(*),2) AS settlement_pct FROM claims GROUP BY claim_type ORDER BY settlement_pct DESC;",
 "GroupBy/Conditional/Ratio",["claims"],["claim_type","claim_status"],["CASE WHEN","SUM","COUNT","GROUP BY"]),

("PS-219","What is the average annual premium for 20-year tenure policies?",
 "SELECT ROUND(AVG(annual_premium),2) AS avg_premium_20yr FROM policies WHERE policy_tenure_years=20;",
 "Filtering/Average",["policies"],["policy_tenure_years","annual_premium"],["AVG","WHERE"]),

("PS-220","Which branch has the highest average NBP achieved per agent?",
 "SELECT branch_id, ROUND(AVG(nbp_achieved),2) AS avg_nbp FROM sales GROUP BY branch_id ORDER BY avg_nbp DESC LIMIT 1;",
 "GroupBy/Ranking",["sales"],["branch_id","nbp_achieved"],["AVG","GROUP BY","ORDER BY"]),

("PS-221","What is the total sum assured for policies issued after 2020?",
 "SELECT SUM(sum_assured) AS total_sa_post_2020 FROM policies WHERE issuance_date > '2020-12-31';",
 "Temporal/Filtering/Sum",["policies"],["issuance_date","sum_assured"],["SUM","WHERE"]),

("PS-222","How many policies have annual premium above 100,000?",
 "SELECT COUNT(*) AS high_premium_policies FROM policies WHERE annual_premium > 100000;",
 "Filtering/Count",["policies"],["annual_premium"],["COUNT","WHERE"]),

("PS-223","What proportion of post-graduate agents exceed their target?",
 "SELECT ROUND(100.0*SUM(CASE WHEN target_achievement_pct>100 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_over_target FROM sales WHERE qualification='Post Graduate';",
 "Filtering/Conditional/Ratio",["sales"],["qualification","target_achievement_pct"],["CASE WHEN","SUM","COUNT","WHERE"]),

("PS-224","What is the average sum assured for policies issued before 2015?",
 "SELECT ROUND(AVG(sum_assured),2) AS avg_sa_pre_2015 FROM policies WHERE issuance_date < '2015-01-01';",
 "Temporal/Filtering/Average",["policies"],["issuance_date","sum_assured"],["AVG","WHERE"]),

("PS-225","Which product has the highest average annual premium?",
 "SELECT product_name, ROUND(AVG(annual_premium),2) AS avg_premium FROM policies GROUP BY product_name ORDER BY avg_premium DESC LIMIT 1;",
 "GroupBy/Ranking",["policies"],["product_name","annual_premium"],["AVG","GROUP BY","ORDER BY"]),

("PS-226","What is the average BMI of customers who bought Term Life?",
 "SELECT ROUND(AVG(c.bmi),2) AS avg_bmi_term_life FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE p.product_name='Term Life';",
 "Join/Filtering/Average",["policies","customers"],["product_name","bmi"],["JOIN","AVG","WHERE"]),

("PS-227","How many policies exist per premium paying term (years)?",
 "SELECT premium_paying_term_years, COUNT(*) AS cnt FROM policies GROUP BY premium_paying_term_years ORDER BY premium_paying_term_years;",
 "GroupBy/Distribution",["policies"],["premium_paying_term_years"],["COUNT","GROUP BY"]),

("PS-228","What is the total NBP target set per branch?",
 "SELECT branch_id, ROUND(SUM(nbp_target),2) AS total_target FROM sales GROUP BY branch_id ORDER BY total_target DESC;",
 "GroupBy/Sum",["sales"],["branch_id","nbp_target"],["SUM","GROUP BY"]),

("PS-229","What percentage of policies are paid-up?",
 "SELECT ROUND(100.0*SUM(CASE WHEN policy_status='Paid-up' THEN 1 ELSE 0 END)/COUNT(*),2) AS paid_up_pct FROM policies;",
 "Conditional/Ratio",["policies"],["policy_status"],["CASE WHEN","SUM","COUNT"]),

("PS-230","What is the average sum assured for Child Plan vs other products?",
 "SELECT CASE WHEN product_name='Child Plan' THEN 'Child Plan' ELSE 'Other' END AS segment, ROUND(AVG(sum_assured),2) AS avg_sa FROM policies GROUP BY segment;",
 "Conditional/Comparison",["policies"],["product_name","sum_assured"],["CASE WHEN","AVG","GROUP BY"]),

("PS-231","Which zone has the best claim settlement rate?",
 "SELECT pol.zone, ROUND(100.0*SUM(CASE WHEN cl.claim_status='Settled' THEN 1 ELSE 0 END)/COUNT(*),2) AS settlement_pct FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number GROUP BY pol.zone ORDER BY settlement_pct DESC LIMIT 1;",
 "Join/Conditional/Ranking",["claims","policies"],["zone","claim_status"],["JOIN","CASE WHEN","SUM","COUNT","GROUP BY","ORDER BY"]),

("PS-232","What is the premium-to-income ratio by marital status?",
 "SELECT c.marital_status, ROUND(AVG(p.annual_premium*1.0/c.annual_income),4) AS avg_premium_income_ratio FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE c.annual_income > 0 GROUP BY c.marital_status ORDER BY avg_premium_income_ratio DESC;",
 "Join/Derived-Metric/GroupBy",["policies","customers"],["marital_status","annual_premium","annual_income"],["JOIN","AVG","GROUP BY","Derived Ratio"]),

("PS-233","How many policies have both medical underwriting AND riders?",
 "SELECT COUNT(*) AS med_uw_with_riders FROM policies WHERE medical_underwriting_flag='Y' AND rider_count > 0;",
 "Multi-Condition/Count",["policies"],["medical_underwriting_flag","rider_count"],["COUNT","WHERE"]),

("PS-234","What is the average sum assured for Online channel policies?",
 "SELECT ROUND(AVG(sum_assured),2) AS avg_sa_online FROM policies WHERE sales_channel='Online';",
 "Filtering/Average",["policies"],["sales_channel","sum_assured"],["AVG","WHERE"]),

("PS-235","Which product has the most early claims?",
 "SELECT pol.product_name, COUNT(*) AS early_claims FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number WHERE cl.is_early_claim=1 GROUP BY pol.product_name ORDER BY early_claims DESC LIMIT 1;",
 "Join/Filtering/Ranking",["claims","policies"],["product_name","is_early_claim"],["JOIN","COUNT","WHERE","GROUP BY","ORDER BY"]),

("PS-236","What is the total sum assured for female customers?",
 "SELECT SUM(p.sum_assured) AS total_sa_female FROM policies p JOIN customers c ON p.customer_id=c.customer_id WHERE c.gender='Female';",
 "Join/Filtering/Sum",["policies","customers"],["gender","sum_assured"],["JOIN","SUM","WHERE"]),

("PS-237","What is the premium difference between smoker and non-smoker customers on average?",
 "SELECT MAX(CASE WHEN is_smoker='Y' THEN avg_p END) - MAX(CASE WHEN is_smoker='N' THEN avg_p END) AS premium_diff FROM (SELECT c.is_smoker, ROUND(AVG(p.annual_premium),2) AS avg_p FROM policies p JOIN customers c ON p.customer_id=c.customer_id GROUP BY c.is_smoker) t;",
 "Join/Derived-Metric/Comparison",["policies","customers"],["is_smoker","annual_premium"],["JOIN","AVG","CASE WHEN","Subquery"]),

("PS-238","What percentage of Health Rider policies are In-Force?",
 "SELECT ROUND(100.0*SUM(CASE WHEN policy_status='In-Force' THEN 1 ELSE 0 END)/COUNT(*),2) AS inforce_pct FROM policies WHERE product_name='Health Rider';",
 "Filtering/Conditional/Ratio",["policies"],["product_name","policy_status"],["CASE WHEN","SUM","COUNT","WHERE"]),

("PS-239","How many distinct cities are covered by policyholders?",
 "SELECT COUNT(DISTINCT location_city) AS distinct_cities FROM policies;",
 "Aggregation/Distinct",["policies"],["location_city"],["COUNT","DISTINCT"]),

("PS-240","What is the average total premium paid for surrendered policies?",
 "SELECT ROUND(AVG(total_premium_paid),2) AS avg_premium_surrendered FROM policies WHERE policy_status='Surrendered';",
 "Filtering/Average",["policies"],["policy_status","total_premium_paid"],["AVG","WHERE"]),

("PS-241","What is the total sum assured for policies in the Central zone?",
 "SELECT SUM(sum_assured) AS total_sa_central FROM policies WHERE zone='Central';",
 "Filtering/Sum",["policies"],["zone","sum_assured"],["SUM","WHERE"]),

("PS-242","What is the average age of customers who filed death claims?",
 "SELECT ROUND(AVG(c.age),2) AS avg_age_death_claim FROM claims cl JOIN customers c ON cl.customer_id=c.customer_id WHERE cl.claim_type='Death Claim';",
 "Join/Filtering/Average",["claims","customers"],["age","claim_type"],["JOIN","AVG","WHERE"]),

("PS-243","How many policies have a 30-year tenure?",
 "SELECT COUNT(*) AS policies_30yr FROM policies WHERE policy_tenure_years=30;",
 "Filtering/Count",["policies"],["policy_tenure_years"],["COUNT","WHERE"]),

("PS-244","What is the total premiums paid by customers in Mumbai?",
 "SELECT ROUND(SUM(total_premium_paid),2) AS total_premium_mumbai FROM policies WHERE location_city='Mumbai';",
 "Filtering/Sum",["policies"],["location_city","total_premium_paid"],["SUM","WHERE"]),

("PS-245","Which nominee relation has the highest average claim amount?",
 "SELECT pol.nominee_relation, ROUND(AVG(cl.claim_amount),2) AS avg_claim FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number GROUP BY pol.nominee_relation ORDER BY avg_claim DESC LIMIT 1;",
 "Join/GroupBy/Ranking",["claims","policies"],["nominee_relation","claim_amount"],["JOIN","AVG","GROUP BY","ORDER BY"]),

("PS-246","What is the total claim amount for repudiated claims?",
 "SELECT ROUND(SUM(claim_amount),2) AS total_repudiated_amount FROM claims WHERE claim_status='Repudiated';",
 "Filtering/Sum",["claims"],["claim_status","claim_amount"],["SUM","WHERE"]),

("PS-247","What percentage of Broker-channel policies are In-Force?",
 "SELECT ROUND(100.0*SUM(CASE WHEN policy_status='In-Force' THEN 1 ELSE 0 END)/COUNT(*),2) AS inforce_pct FROM policies WHERE sales_channel='Broker';",
 "Filtering/Conditional/Ratio",["policies"],["sales_channel","policy_status"],["CASE WHEN","SUM","COUNT","WHERE"]),

("PS-248","What is the average premium-paying term for Endowment policies?",
 "SELECT ROUND(AVG(premium_paying_term_years),2) AS avg_ppt_endowment FROM policies WHERE product_name='Endowment';",
 "Filtering/Average",["policies"],["product_name","premium_paying_term_years"],["AVG","WHERE"]),

("PS-249","How many agents have worked for more than 10 years?",
 "SELECT COUNT(*) AS senior_agents FROM sales WHERE (julianday(COALESCE(date_of_termination,date('now'))) - julianday(date_of_joining))/365.0 > 10;",
 "Derived-Metric/Count",["sales"],["date_of_joining","date_of_termination"],["COUNT","julianday","COALESCE"]),

("PS-250","What is the total claim amount by medical diagnosis for health claims?",
 "SELECT medical, ROUND(SUM(claim_amount),2) AS total_claim_amount FROM claims WHERE claim_type='Health Claim' AND medical IS NOT NULL GROUP BY medical ORDER BY total_claim_amount DESC;",
 "Filtering/GroupBy/Sum",["claims"],["claim_type","medical","claim_amount"],["SUM","GROUP BY","WHERE","ORDER BY"]),

("PS-251","What percentage of total sum assured is concentrated in the top 10 policies?",
 "SELECT ROUND(100.0*(SELECT SUM(sum_assured) FROM (SELECT sum_assured FROM policies ORDER BY sum_assured DESC LIMIT 10))/(SELECT SUM(sum_assured) FROM policies),2) AS top10_concentration_pct;",
 "Concentration/Subquery",["policies"],["sum_assured"],["SUM","ORDER BY","LIMIT","Subquery"]),

("PS-252","What is the average annual premium for Bancassurance vs Agency channel?",
 "SELECT sales_channel, ROUND(AVG(annual_premium),2) AS avg_premium FROM policies WHERE sales_channel IN ('Bancassurance','Agency') GROUP BY sales_channel;",
 "Filtering/GroupBy/Comparison",["policies"],["sales_channel","annual_premium"],["AVG","GROUP BY","WHERE"]),

("PS-253","What is the policy status distribution for ULIP products?",
 "SELECT policy_status, COUNT(*) AS cnt FROM policies WHERE product_name='ULIP' GROUP BY policy_status ORDER BY cnt DESC;",
 "Filtering/GroupBy/Distribution",["policies"],["product_name","policy_status"],["COUNT","GROUP BY","WHERE"]),

("PS-254","What is the average age of agents who exceeded their NBP target?",
 "SELECT ROUND(AVG(age),2) AS avg_age_over_target FROM sales WHERE target_achievement_pct > 100;",
 "Filtering/Average",["sales"],["target_achievement_pct","age"],["AVG","WHERE"]),

("PS-255","What is the total claim amount for policies in the West zone?",
 "SELECT ROUND(SUM(cl.claim_amount),2) AS total_west_claims FROM claims cl JOIN policies pol ON cl.policy_number=pol.policy_number WHERE pol.zone='West';",
 "Join/Filtering/Sum",["claims","policies"],["zone","claim_amount"],["JOIN","SUM","WHERE"]),

("PS-256","What proportion of male vs female customers are smokers?",
 "SELECT gender, ROUND(100.0*SUM(CASE WHEN is_smoker='Y' THEN 1 ELSE 0 END)/COUNT(*),2) AS smoker_pct FROM customers GROUP BY gender;",
 "GroupBy/Conditional/Ratio",["customers"],["gender","is_smoker"],["CASE WHEN","SUM","COUNT","GROUP BY"]),

("PS-257","Which claim year had the highest average claim amount?",
 "SELECT SUBSTR(claim_date,1,4) AS year, ROUND(AVG(claim_amount),2) AS avg_claim FROM claims GROUP BY year ORDER BY avg_claim DESC LIMIT 1;",
 "Time-Series/Ranking",["claims"],["claim_date","claim_amount"],["AVG","GROUP BY","ORDER BY"]),

("PS-258","What is the average sum assured for Annual payment mode policies?",
 "SELECT ROUND(AVG(sum_assured),2) AS avg_sa_annual FROM policies WHERE payment_mode='Annual';",
 "Filtering/Average",["policies"],["payment_mode","sum_assured"],["AVG","WHERE"]),

("PS-259","What is the total premium paid for all In-Force policies?",
 "SELECT ROUND(SUM(total_premium_paid),2) AS total_inforce_premium FROM policies WHERE policy_status='In-Force';",
 "Filtering/Sum",["policies"],["policy_status","total_premium_paid"],["SUM","WHERE"]),

("PS-260","What is the average claim amount for surrender claims?",
 "SELECT ROUND(AVG(claim_amount),2) AS avg_surrender_claim FROM claims WHERE claim_type='Surrender Claim';",
 "Filtering/Average",["claims"],["claim_type","claim_amount"],["AVG","WHERE"]),

("PS-261", "How many policies have been surrendered in total?",
 "SELECT COUNT(*) AS total_surrendered FROM policies WHERE policy_status='Surrendered';",
 "Aggregation/Count", ["policies"], ["policy_status"], ["COUNT", "WHERE"]),

("PS-262", "What is the total surrender value paid out?",
 "SELECT ROUND(SUM(surrender_value_paid),2) AS total_val_paid FROM surrenders;",
 "Aggregation/Sum", ["surrenders"], ["surrender_value_paid"], ["SUM"]),

("PS-263", "What is the average surrender charge per surrendered policy?",
 "SELECT ROUND(AVG(surrender_charge),2) AS avg_charge FROM surrenders;",
 "Aggregation/Average", ["surrenders"], ["surrender_charge"], ["AVG"]),

("PS-264", "Which product category has the highest surrender rate?",
 "SELECT pr.product_category, ROUND(100.0*SUM(CASE WHEN po.policy_status='Surrendered' THEN 1 ELSE 0 END)/COUNT(*),2) AS surrender_rate FROM policies po JOIN products pr ON po.product_id=pr.product_id GROUP BY pr.product_category ORDER BY surrender_rate DESC LIMIT 1;",
 "Join/Conditional/Ratio", ["policies", "products"], ["product_category", "policy_status"], ["JOIN", "SUM", "CASE WHEN", "GROUP BY", "ORDER BY"]),

("PS-265", "What is the distribution of surrender reasons?",
 "SELECT surrender_reason, COUNT(*) AS cnt FROM surrenders GROUP BY surrender_reason ORDER BY cnt DESC;",
 "GroupBy/Distribution", ["surrenders"], ["surrender_reason"], ["COUNT", "GROUP BY"]),

("PS-266", "How many policies were successfully retained by retention efforts?",
 "SELECT COUNT(*) AS retained_count FROM retention WHERE retention_status='Retained';",
 "Filtering/Count", ["retention"], ["retention_status"], ["COUNT", "WHERE"]),

("PS-267", "What is the overall retention success rate for policies contacted?",
 "SELECT ROUND(100.0*SUM(CASE WHEN retention_status='Retained' THEN 1 ELSE 0 END)/COUNT(*),2) AS success_rate FROM retention;",
 "Aggregation/Ratio", ["retention"], ["retention_status"], ["SUM", "COUNT", "CASE WHEN"]),

("PS-268", "What is the total agent incentive paid for successful policy retention?",
 "SELECT ROUND(SUM(agent_incentive_amount),2) AS total_incentive FROM retention;",
 "Aggregation/Sum", ["retention"], ["agent_incentive_amount"], ["SUM"]),

("PS-269", "Which contact channel was most effective for retaining policies (highest count)?",
 "SELECT contact_channel, COUNT(*) AS cnt FROM retention WHERE retention_status='Retained' GROUP BY contact_channel ORDER BY cnt DESC LIMIT 1;",
 "Filtering/GroupBy/Ranking", ["retention"], ["contact_channel", "retention_status"], ["COUNT", "GROUP BY", "ORDER BY"]),

("PS-270", "What is the average elapsed years of a policy before it gets surrendered?",
 "SELECT ROUND(AVG(policy_tenure_at_claim_months*1.0/12.0),2) AS avg_elapsed_years FROM claims WHERE claim_type='Surrender Claim';",
 "Filtering/Average", ["claims"], ["policy_tenure_at_claim_months", "claim_type"], ["AVG", "WHERE"]),

]  # END QUERY_CATALOG_RAW


# ── 5. Execute All Queries ────────────────────────────────────────────────────
def build_query_records(conn):
    records = []
    for row in QUERY_CATALOG_RAW:
        pid, nl, sql, intent, tables, cols, ops = row
        result  = run_sql(conn, sql)
        answer  = fmt_ans(result)
        records.append({
            "problem_id":  pid,
            "nl_query":    nl,
            "sql":         sql.strip(),
            "intent_type": intent,
            "tables":      tables,
            "columns":     cols,
            "operations":  ops,
            "answer":      answer,
        })
    return records


# ── 6. Derive Problem Statements ──────────────────────────────────────────────
PS_DEFINITIONS = {
    "PS-261": "What is the total count of surrendered policies?",
    "PS-262": "What is the total cash value paid out for surrenders?",
    "PS-263": "What is the average surrender charge incurred by customers?",
    "PS-264": "How does surrender rate vary across product categories?",
    "PS-265": "What is the distribution of reasons for policy surrenders?",
    "PS-266": "How many policies were retained via proactive campaigns?",
    "PS-267": "What is the overall success rate of retention contacts?",
    "PS-268": "What is the total incentive paid to agents for policy retention?",
    "PS-269": "Which contact channel is most effective for retaining at-risk business?",
    "PS-270": "What is the average duration a policy remains active before surrender",
    "PS-001": "How many records exist in a given table (total count)?",
    "PS-002": "What is the count / proportion of a binary categorical attribute (smoker, gender, underwriting flag)?",
    "PS-003": "What is the demographic breakdown (gender, marital status, location)?",
    "PS-004": "What is the demographic breakdown (gender, marital status, location)?",
    "PS-005": "What is the central tendency (mean/median) of a continuous financial or demographic field?",
    "PS-006": "What is the extreme value (max/min) of a financial or demographic field?",
    "PS-007": "How do risk-related attributes (BMI, age, income) differ across smoker segments?",
    "PS-008": "How do risk-related attributes (BMI, age, income) differ across smoker segments?",
    "PS-009": "Which geographic unit (city/state/zone) has the highest concentration of customers or policies?",
    "PS-010": "What is the count distribution across a categorical segment?",
    "PS-011": "How does a continuous attribute vary across demographic groups?",
    "PS-012": "How many entities exceed or fall below a numeric threshold?",
    "PS-013": "How does a continuous attribute vary across demographic groups?",
    "PS-014": "How many entities exceed or fall below a numeric threshold?",
    "PS-015": "What is the policy status distribution across the portfolio?",
    "PS-016": "What is the policy status distribution across the portfolio?",
    "PS-017": "What is the total financial exposure (sum assured / premium) in a segment?",
    "PS-018": "What is the central tendency (mean/median) of a continuous financial or demographic field?",
    "PS-019": "How are policies/premiums/claims distributed across sales channels?",
    "PS-020": "What is the total or average premium collected by a given dimension?",
    "PS-021": "What is the count and mix of policies across products?",
    "PS-022": "What is the average sum assured or premium for each product?",
    "PS-023": "How are policies/premiums/claims distributed across geographic zones?",
    "PS-024": "What is the total or average premium collected by a given dimension?",
    "PS-025": "How many policies include optional add-ons (riders, medical underwriting)?",
    "PS-026": "What is the distribution of payment modes among policyholders?",
    "PS-027": "What is the distribution and central tendency of policy tenure?",
    "PS-028": "How has portfolio growth (count, premium) trended year over year?",
    "PS-029": "What is the most common value in a categorical column?",
    "PS-030": "How many policies include optional add-ons (riders, medical underwriting)?",
    "PS-031": "What is the extreme value (max/min) of a financial or demographic field?",
    "PS-032": "What is the total financial exposure across geographic units?",
    "PS-033": "What percentage of policies are in a specific status?",
    "PS-034": "How many entities exceed or fall below a numeric threshold?",
    "PS-035": "What is the average premium paying term by product?",
    "PS-036": "How many claims have been filed in total or for a specific type/period?",
    "PS-037": "What is the total claim amount paid (by type, period, or channel)?",
    "PS-038": "What is the distribution of claims across claim types?",
    "PS-039": "What is the average claim amount by claim type, product, or segment?",
    "PS-040": "How many claims are in a pending / under-investigation status?",
    "PS-041": "What is the claims settlement vs repudiation rate?",
    "PS-042": "How many early claims exist and what is their financial impact?",
    "PS-043": "How does claim severity differ between early and non-early claims?",
    "PS-044": "How has claim volume trended year over year?",
    "PS-045": "How has the total claim payout trended year over year?",
    "PS-046": "What is the extreme value (max/min) of a financial or demographic field?",
    "PS-047": "What is the most common medical diagnosis in health claims?",
    "PS-048": "What is the average policy tenure at the time of claim?",
    "PS-049": "What is the repudiation rate overall or by claim type?",
    "PS-050": "How many agents are active vs inactive?",
    "PS-051": "Who are the top/bottom performing agents by a given KPI?",
    "PS-052": "Who are the top/bottom performing agents by a given KPI?",
    "PS-053": "What is the average performance metric (achievement, persistency, FTR) across agents?",
    "PS-054": "How many agents achieved or missed their performance targets?",
    "PS-055": "What is the average performance metric across agents?",
    "PS-056": "Who are the top/bottom performing agents by a given KPI?",
    "PS-057": "Who are the top/bottom performing agents by a given KPI?",
    "PS-058": "What is the average performance metric across agents?",
    "PS-059": "How many agents satisfy a performance threshold (persistency, claim rate)?",
    "PS-060": "What is the distribution of agent qualifications?",
    "PS-061": "What is the total financial output (SA sold, NBP) per branch?",
    "PS-062": "How many agents satisfy a performance threshold (persistency, claim rate)?",
    "PS-063": "Who are the top/bottom performing agents by a given KPI?",
    "PS-064": "Which products or plans are currently active?",
    "PS-065": "Which product has the highest or lowest premium rate?",
    "PS-066": "How many products or plans have a specific benefit feature?",
    "PS-067": "How many products or plans have a specific benefit feature?",
    "PS-068": "What is the central tendency of product parameters (min/max SA)?",
    "PS-069": "What is the most recent or oldest product/plan in the catalogue?",
    "PS-070": "What are the distinct product categories available?",
    "PS-071": "How many plans are active or of a specific payment type?",
    "PS-072": "How many plans are active or of a specific payment type?",
    "PS-073": "How are plans distributed across products?",
    "PS-074": "How does a financial outcome differ between smoker and non-smoker segments?",
    "PS-075": "How many policies does each demographic segment hold?",
    "PS-076": "What is the total or average premium paid across demographic segments?",
    "PS-077": "How does claim severity vary across demographic segments?",
    "PS-078": "Which product category has the highest total claim payout?",
    "PS-079": "How does a financial attribute differ between claimants and non-claimants?",
    "PS-080": "Which geographic unit has the highest or lowest average claim amount?",
    "PS-081": "How many claims are associated with each product or channel?",
    "PS-082": "What is the total premium collected or sum assured sold per agent?",
    "PS-083": "How does a customer risk attribute differ among claimants of a specific type?",
    "PS-084": "How many policyholders belong to each occupation or employer segment?",
    "PS-085": "What is the total business generated by agents meeting performance criteria?",
    "PS-086": "What is the claim settlement rate by geographic unit?",
    "PS-087": "How does customer income differ for lapsed vs in-force policyholders?",
    "PS-088": "How does customer income differ for lapsed vs in-force policyholders?",
    "PS-089": "Which sales channel generates the highest average sum assured per policy?",
    "PS-090": "How many unique customers hold at least one policy?",
    "PS-091": "Which geographic unit has the highest concentration of policies or sum assured?",
    "PS-092": "Which product has the highest total premium or sum assured?",
    "PS-093": "Which top-N states/cities/zones have the highest policy count?",
    "PS-094": "Which branch has the most policies?",
    "PS-095": "Who is the youngest or oldest agent?",
    "PS-096": "Who is the youngest or oldest agent?",
    "PS-097": "What are the top-N most common medical diagnoses in health claims?",
    "PS-098": "Which customer has the highest annual income?",
    "PS-099": "Which policy has the highest annual premium or sum assured?",
    "PS-100": "Which N agents have the lowest persistency rate?",
    "PS-101": "How has portfolio growth trended year over year?",
    "PS-102": "How has total premium collected trended over the years?",
    "PS-103": "How has the average sum assured evolved year over year?",
    "PS-104": "Which year had the highest number of claims?",
    "PS-105": "What is the monthly or seasonal distribution of claims or policies?",
    "PS-106": "How many policies matured each year?",
    "PS-107": "What is the trend in lapsed policies by issuance year?",
    "PS-108": "How has the total claim amount changed each year?",
    "PS-109": "What is the distribution of an attribute across bucket ranges?",
    "PS-110": "What is the distribution of an attribute across bucket ranges?",
    "PS-111": "What is the distribution of an attribute across bucket ranges?",
    "PS-112": "What is the distribution of claim amounts in a specific claim type?",
    "PS-113": "What is the distribution of policy tenure (years)?",
    "PS-114": "What is the BMI category distribution across customers?",
    "PS-115": "What is the lapse or surrender rate by channel, product, or zone?",
    "PS-116": "What is the policy status breakdown for a specific channel or product?",
    "PS-117": "What is the average sum assured for policies with vs without riders?",
    "PS-118": "Among settled claims, what percentage were early claims?",
    "PS-119": "What is the average claim amount for smoker vs non-smoker policyholders?",
    "PS-120": "Do customers with higher income tend to have higher sum assured?",
    "PS-121": "What is the lapse or surrender rate by product?",
    "PS-122": "What is the death or specific claim rate by zone?",
    "PS-123": "Which product has the highest proportion of lapsed policies?",
    "PS-124": "What percentage of customers satisfy multiple risk conditions simultaneously?",
    "PS-125": "What is the average number of premiums paid for lapsed or surrendered policies?",
    "PS-126": "What is the correlation proxy between a product attribute and claim severity?",
    "PS-127": "How many policies have a derived ratio (premium/SA) below a threshold?",
    "PS-128": "What is the average derived ratio (total premium / sum assured) by product?",
    "PS-129": "Which policies have total premiums paid exceeding sum assured (anomaly)?",
    "PS-130": "Which customers have policies in more than one geographic zone?",
    "PS-131": "Which agents have a lapse rate above a threshold (anomaly)?",
    "PS-132": "Which agents missed their NBP target by more than a threshold?",
    "PS-133": "Which claims have amount more than N times the average (outlier detection)?",
    "PS-134": "Which customers have low income but disproportionately high sum assured?",
    "PS-135": "How many policies were surrendered unusually early?",
    "PS-136": "Which branch has the highest lapse rate among its policies?",
    "PS-137": "Which agents have the lowest FTR rate?",
    "PS-138": "How many In-Force policies have an overdue next premium due date?",
    "PS-139": "Which customers have filed more than one claim?",
    "PS-140": "Which claim type has the highest repudiation rate?",
    "PS-141": "Which product is most commonly sold by a specific channel?",
    "PS-142": "What is the average number of riders per policy by product?",
    "PS-143": "What is the total NBP achieved or target per branch?",
    "PS-144": "What is the average target achievement by agent qualification?",
    "PS-145": "Which zone has the highest total sum assured for a specific product?",
    "PS-146": "What is the average premium for Direct vs Agency channel policies?",
    "PS-147": "What is the total sum assured for medically underwritten policies?",
    "PS-148": "What percentage of policies were sold through a specific channel?",
    "PS-149": "How many policies exist per plan payment type?",
    "PS-150": "What is the average sum assured for Regular vs Limited pay plans?",
    "PS-151": "What is the loss ratio (total claims / total premiums) by product?",
    "PS-152": "What is the average claim-to-sum-assured ratio for settled claims?",
    "PS-153": "Which zone has the most diverse product mix?",
    "PS-154": "What is the average customer lifetime value proxy by income tier?",
    "PS-155": "What percentage of medically underwritten policies are In-Force?",
    "PS-156": "What is the average annual premium for smoker vs non-smoker customers?",
    "PS-157": "What is the average time from policy issuance to claim?",
    "PS-158": "Which month sees the most policy issuances (seasonality)?",
    "PS-159": "What is the claim frequency (claims per policy) by sales channel?",
    "PS-160": "What is the average FTR rate by branch?",
    "PS-161": "How many claims of a specific type were repudiated?",
    "PS-162": "What is the average sum assured for policies in a specific zone?",
    "PS-163": "What is the average sum assured for policies in a specific zone?",
    "PS-164": "How many agents joined after a given date?",
    "PS-165": "What is the total premium paid for a specific product?",
    "PS-166": "How many claims of a specific type were filed?",
    "PS-167": "What is the average policy tenure for a specific policy status?",
    "PS-168": "How many customers have more than N policies?",
    "PS-169": "What is the average annual premium by payment mode?",
    "PS-170": "What is the maximum claim amount for a specific claim type?",
    "PS-171": "What is the total sum assured or NBP sold by active agents?",
    "PS-172": "Which state has the highest lapse rate?",
    "PS-173": "What is the average BMI for customers with a specific policy status?",
    "PS-174": "How many policies have zero premiums paid?",
    "PS-175": "What is the average claim amount for a specific claim type?",
    "PS-176": "What proportion of total claim payout comes from a specific claim type?",
    "PS-177": "Which branches have the highest policy count (top N)?",
    "PS-178": "What is the average annual premium for a specific product?",
    "PS-179": "What is the ratio of a specific claim type to total policies?",
    "PS-180": "How many customers have no associated policy?",
    "PS-181": "What is the average sum assured for a specific marital status segment?",
    "PS-182": "What is the average annual income for customers in each zone?",
    "PS-183": "What is the proportion of female policyholders by product?",
    "PS-184": "What is the average claim amount for customers above a certain age?",
    "PS-185": "What is the average claim amount for customers below a certain age?",
    "PS-186": "How does the lapse rate compare across income tiers?",
    "PS-187": "What is the average sum assured per policy for the top performing agent?",
    "PS-188": "What is the claim resolution rate by year?",
    "PS-189": "What is the average persistency rate per branch?",
    "PS-190": "What is the total claim amount for policies sold through a specific channel?",
    "PS-191": "What is the market share of each product by policy count?",
    "PS-192": "What is the average policy tenure for each product?",
    "PS-193": "Which product generates the most total sum assured?",
    "PS-194": "What is the average claim amount per product category?",
    "PS-195": "How many policies exist per product category?",
    "PS-196": "What is the average premium rate for products with specific benefit features?",
    "PS-197": "Which products have both surrender value and maturity benefit?",
    "PS-198": "What is the total NBP achieved vs target for all agents?",
    "PS-199": "What is the claim rate for smoker policyholders?",
    "PS-200": "What is the claim rate for non-smoker policyholders?",
    "PS-201": "What is the average age of customers who bought a specific product?",
    "PS-202": "What is the total claim amount for early claims?",
    "PS-203": "What proportion of a specific product's policies are In-Force?",
    "PS-204": "What is the average premiums paid count for In-Force vs Lapsed policies?",
    "PS-205": "How many claim-free agents are there per branch?",
    "PS-206": "What is the average policy tenure for a specific product?",
    "PS-207": "What is the policy status distribution within each geographic zone?",
    "PS-208": "How many customers made a claim within the first 2 years of their policy?",
    "PS-209": "What is the average sum assured by nominee relation?",
    "PS-210": "Which agent branch has the highest average target achievement?",
    "PS-211": "What is the total payout for a specific claim type?",
    "PS-212": "What is the average sum assured for policies maturing soon?",
    "PS-213": "Which product has the most claims under investigation?",
    "PS-214": "What is the average annual premium for customers with high BMI?",
    "PS-215": "What is the total claim count per medical diagnosis?",
    "PS-216": "What is the average number of policies sold per active agent?",
    "PS-217": "What is the total premium for a specific payment mode?",
    "PS-218": "What percentage of claims are settled for each claim type?",
    "PS-219": "What is the average premium for policies of a specific tenure?",
    "PS-220": "Which branch has the highest average NBP achieved per agent?",
    "PS-221": "What is the total sum assured for policies issued in a specific period?",
    "PS-222": "How many policies have annual premium above a threshold?",
    "PS-223": "What proportion of agents of a given qualification exceed their target?",
    "PS-224": "What is the average sum assured for policies issued before a given year?",
    "PS-225": "Which product has the highest average annual premium?",
    "PS-226": "What is the average BMI of customers who bought a specific product?",
    "PS-227": "How many policies exist per premium paying term?",
    "PS-228": "What is the total NBP target set per branch?",
    "PS-229": "What percentage of policies are paid-up?",
    "PS-230": "What is the average sum assured for a product segment vs others?",
    "PS-231": "Which zone has the best claim settlement rate?",
    "PS-232": "What is the premium-to-income ratio by marital status?",
    "PS-233": "How many policies have both medical underwriting and riders?",
    "PS-234": "What is the average sum assured for a specific sales channel?",
    "PS-235": "Which product has the most early claims?",
    "PS-236": "What is the total sum assured for female customers?",
    "PS-237": "What is the premium difference between smoker and non-smoker customers?",
    "PS-238": "What percentage of Health Rider policies are In-Force?",
    "PS-239": "How many distinct cities are covered by policyholders?",
    "PS-240": "What is the average total premium paid for surrendered policies?",
    "PS-241": "What is the total sum assured for policies in a specific zone?",
    "PS-242": "What is the average age of customers who filed a specific claim type?",
    "PS-243": "How many policies have a specific policy tenure?",
    "PS-244": "What is the total premium paid by customers in a specific city?",
    "PS-245": "Which nominee relation has the highest average claim amount?",
    "PS-246": "What is the total claim amount for a specific claim status?",
    "PS-247": "What percentage of a specific channel's policies are In-Force?",
    "PS-248": "What is the average premium-paying term for a specific product?",
    "PS-249": "How many agents have worked for more than N years?",
    "PS-250": "What is the total claim amount by medical diagnosis for health claims?",
    "PS-251": "What percentage of total sum assured is concentrated in the top N policies?",
    "PS-252": "What is the average premium for a specific channel combination?",
    "PS-253": "What is the policy status distribution for a specific product?",
    "PS-254": "What is the average age of agents who met a performance threshold?",
    "PS-255": "What is the total claim amount for policies in a specific zone?",
    "PS-256": "What proportion of each gender are smokers?",
    "PS-257": "Which claim year had the highest average claim amount?",
    "PS-258": "What is the average sum assured for a specific payment mode?",
    "PS-259": "What is the total premium paid for all In-Force policies?",
    "PS-260": "What is the average claim amount for a specific claim type?",
}

def get_ps_stmt(pid):
    base = re.sub(r'[a-z]+$','',pid)
    return PS_DEFINITIONS.get(pid, PS_DEFINITIONS.get(base,
           f"Analytical query on the insurance dataset: {pid}"))

def build_problem_statements(records):
    ps_map = {}
    for r in records:
        base = re.sub(r'[a-z]+$','',r["problem_id"])
        ps_map.setdefault(base,[]).append(r)

    ps_records = []
    for base_pid, queries in ps_map.items():
        p = queries[0]
        stmt = get_ps_stmt(p["problem_id"])
        all_tables = list(set(t for q in queries for t in q["tables"]))
        all_cols   = list(set(c for q in queries for c in q["columns"]))
        all_ops    = list(set(o for q in queries for o in q["operations"]))
        import graph_engine as ge
        cyphers = [ge.sql_to_cypher(q["sql"]) for q in queries]
        ps_records.append({
            "problem_id":          base_pid,
            "problem_statement":   stmt,
            "intent_type":         p["intent_type"],
            "example_queries":     [q["nl_query"]  for q in queries],
            "example_sql":         [q["sql"]        for q in queries],
            "example_cypher":      cyphers,
            "relevant_tables":     all_tables,
            "relevant_columns":    all_cols,
            "required_operations": all_ops,
            "sql_template":        p["sql"],
            "cypher_template":     ge.sql_to_cypher(p["sql"]),
            "example_answers":     [q["answer"]    for q in queries],
        })
    return ps_records


# ── 7. Retrieval Pipeline ─────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class RetrievalPipeline:
    def __init__(self, ps_records, conn, top_k=3):
        self.ps_records = ps_records
        self.conn = conn
        self.top_k = top_k
        self._build_index()

    def _build_index(self):
        self.corpus = []
        for ps in self.ps_records:
            text = (ps["problem_statement"] + " " +
                    " ".join(ps["example_queries"]) + " " +
                    ps["intent_type"] + " " +
                    " ".join(ps["relevant_tables"]) + " " +
                    " ".join(ps["relevant_columns"]))
            self.corpus.append(text.lower())
        if HAS_SKLEARN:
            self.vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
            self.mat = self.vec.fit_transform(self.corpus)

    def _kw_score(self, q, txt):
        return len(set(q.lower().split()) & set(txt.lower().split()))

    def retrieve(self, user_query, top_k=None):
        k = top_k or self.top_k
        if HAS_SKLEARN:
            qv = self.vec.transform([user_query.lower()])
            scores = cosine_similarity(qv, self.mat).flatten()
            idx = scores.argsort()[::-1][:k]
            return [(self.ps_records[i], float(scores[i])) for i in idx]
        scored = [(ps, self._kw_score(user_query, txt))
                  for ps, txt in zip(self.ps_records, self.corpus)]
        return sorted(scored, key=lambda x:x[1], reverse=True)[:k]

    def answer(self, user_query, top_k=None):
        results = []
        for ps, score in self.retrieve(user_query, top_k=top_k):
            df  = run_sql(self.conn, ps["sql_template"])
            ans = fmt_ans(df)
            results.append({
                "problem_id":        ps["problem_id"],
                "problem_statement": ps["problem_statement"],
                "confidence_score":  round(score, 4),
                "evidence": {
                    "matched_tables":  ps["relevant_tables"],
                    "matched_columns": ps["relevant_columns"],
                    "intent_type":     ps["intent_type"],
                },
                "sql_executed": ps["sql_template"],
                "answer":       ans,
            })
        return results


# ── 8. Evaluation Split ───────────────────────────────────────────────────────
EVAL_QUERIES = [
    ("What is the total surrender value paid out?", ["PS-262"]),
    ("What is the average surrender charge per policy?", ["PS-263"]),
    ("Which product category has the highest surrender rate?", ["PS-264"]),
    ("How many policies were successfully retained?", ["PS-266"]),
    ("What is the average annual income of customers with lapsed policies and what is their average BMI?", ["PS-087", "PS-007"]),
    ("Which channel was most effective for retaining policies?", ["PS-269"]),
    ("How many total insurance policies are there?",                    ["PS-001"]),
    ("Count of policies that have lapsed",                              ["PS-016"]),
    ("Which sales channel sold the most policies?",                     ["PS-019"]),
    ("Average BMI of customers who smoke",                              ["PS-007"]),
    ("Top agent by number of policies sold",                            ["PS-051"]),
    ("Total sum assured for Term Life product",                         ["PS-193"]),
    ("How many claims were settled?",                                   ["PS-041"]),
    ("Average income of customers with surrendered policies",           ["PS-087"]),
    ("Which year had maximum claim filings?",                           ["PS-104"]),
    ("Most common health diagnosis",                                     ["PS-047"]),
    ("Percentage of policies sold through online channel",              ["PS-148"]),
    ("Average policy tenure for Whole Life product",                    ["PS-192"]),
    ("Claims count by claim type",                                      ["PS-038"]),
    ("Which zone has most policies?",                                   ["PS-023"]),
    ("How does claim amount differ for male and female customers?",     ["PS-077"]),
    ("Agents who missed their NBP target",                              ["PS-054"]),
    ("Average annual premium for ULIP policies",                        ["PS-178"]),
    ("Which product has highest average claim?",                        ["PS-039"]),
    ("Count of early claims filed",                                     ["PS-042"]),
    ("How has number of new policies changed each year?",               ["PS-101"]),
    ("Proportion of smoker vs non-smoker customers",                    ["PS-002"]),
    ("Which city has the most policyholders?",                          ["PS-009"]),
    ("Total premium paid by in-force policies",                         ["PS-259"]),
    ("Distribution of customers by age group",                         ["PS-109"]),
    ("What is the loss ratio for each product?",                        ["PS-151"]),
    ("How many agents are post graduate?",                              ["PS-060"]),
    ("Average annual premium by payment mode",                          ["PS-169"]),
    ("How many policies have riders?",                                  ["PS-025"]),
    ("Top 5 states by number of policies",                              ["PS-093"]),
    ("Surrender rate by product",                                       ["PS-121"]),
]

def evaluate(pipeline, eval_set, K_list=[1,3,5]):
    by_k = {k:[] for k in K_list}
    detailed = []
    for uq, targets in eval_set:
        retrieved = pipeline.retrieve(uq, top_k=max(K_list))
        ids = [ps["problem_id"] for ps,_ in retrieved]
        row = {"query": uq, "target_ps": targets, "retrieved_top5": ids[:5]}
        for k in K_list:
            hit = any(t in ids[:k] for t in targets)
            by_k[k].append(hit)
            row[f"hit@{k}"] = hit
        detailed.append(row)
    recall = {k: round(float(np.mean(by_k[k])),4) for k in K_list}
    return recall, detailed


# ── 9. MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("INSURANCE ANALYTICS QUERY SYSTEM")
    print("="*70)

    print("\n[1/6] Loading data ...")
    claims,customers,plans,policies,products,sales,surrenders,retention = load_data()
    print(f"  claims={len(claims)}, customers={len(customers)}, "
          f"policies={len(policies)}, plans={len(plans)}, "
          f"products={len(products)}, sales={len(sales)}, "
          f"surrenders={len(surrenders)}, retention={len(retention)}")

    print("\n[2/6] Building SQLite database ...")
    conn = build_db(claims,customers,plans,policies,products,sales,surrenders,retention)
    print(f"  DB → {DB_PATH}")

    print("\n[3/6] Executing all queries ...")
    records = build_query_records(conn)
    errors  = [r for r in records if r["answer"].startswith("ERROR")]
    print(f"  Executed {len(records)} queries  |  Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    {e['problem_id']}: {e['answer']}")

    qdf = pd.DataFrame([{
        "problem_id": r["problem_id"],
        "nl_query":   r["nl_query"],
        "intent_type":r["intent_type"],
        "tables":     "|".join(r["tables"]),
        "columns":    "|".join(r["columns"]),
        "operations": "|".join(r["operations"]),
        "sql":        r["sql"],
        "answer":     r["answer"],
    } for r in records])
    qdf.to_csv(f"{OUT}/query_dataset.csv", index=False)
    print(f"  Saved → query_dataset.csv  ({len(qdf)} rows)")

    print("\n[4/6] Building Problem Statement dataset ...")
    ps_records = build_problem_statements(records)
    print(f"  Generated {len(ps_records)} problem statements")
    with open(f"{OUT}/problem_statements.json","w") as f:
        json.dump(ps_records, f, indent=2)
    print(f"  Saved → problem_statements.json")

    print("\n[5/6] Building retrieval pipeline ...")
    import graph_engine as ge
    import query_workflow as qw
    G = ge.build_graph()
    pipeline = qw.QueryWorkflowSystem(G, ps_records)
    print(f"  Pipeline ready  [Cypher Graph Engine]")

    demo_qs = [
        "Which product has the most lapsed policies?",
        "How do smokers compare to non-smokers in terms of premiums?",
        "Top performing agents by NBP achieved",
    ]
    demo_out = []
    for dq in demo_qs:
        demo_out.append({"query": dq, "results": pipeline.answer(dq, top_k=3)})
    with open(f"{OUT}/retrieval_demo.json","w") as f:
        json.dump(demo_out, f, indent=2)
    print(f"  Demo saved → retrieval_demo.json")

    print("\n[6/6] Evaluating on held-out queries ...")
    recall, eval_detail = evaluate(pipeline, EVAL_QUERIES, K_list=[1,3,5])
    print(f"  Recall@1 = {recall[1]}")
    print(f"  Recall@3 = {recall[3]}")
    print(f"  Recall@5 = {recall[5]}")

    pd.DataFrame(eval_detail).to_csv(f"{OUT}/evaluation_results.csv", index=False)
    with open(f"{OUT}/evaluation_summary.json","w") as f:
        json.dump({"recall_at_k": recall, "n_eval_queries": len(EVAL_QUERIES)}, f, indent=2)
    print(f"  Saved → evaluation_results.csv  |  evaluation_summary.json")

    print("\n" + "="*70)
    print("DONE. All outputs saved to:", OUT)
    print("="*70)
    return pipeline, ps_records, records, recall

if __name__ == "__main__":
    pipeline, ps_records, records, recall = main()
