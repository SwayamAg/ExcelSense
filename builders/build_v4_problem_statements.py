"""
build_v4_problem_statements.py
==============================
Generates and normalizes problem statements to strictly match the 5 production
tables: Policy_Details, Owner_Details, Persistency_Details, Sales_Details, Claims_Details.
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
PS_DIR = os.path.join(ROOT_DIR, "data", "problem_statements")
os.makedirs(PS_DIR, exist_ok=True)

TABLE_MAP = {
    "policies": "Policy_Details",
    "policy_data": "Policy_Details",
    "policy": "Policy_Details",
    "customers": "Owner_Details",
    "customer_data": "Owner_Details",
    "customer": "Owner_Details",
    "sales": "Sales_Details",
    "sales_info": "Sales_Details",
    "agent": "Sales_Details",
    "agents": "Sales_Details",
    "claims": "Claims_Details",
    "claims_data": "Claims_Details",
    "claim": "Claims_Details",
    "retention": "Persistency_Details",
    "retention_data": "Persistency_Details",
    "persistency": "Persistency_Details",
    "surrenders": "Policy_Details",
    "surrender_data": "Policy_Details",
    "products": "Policy_Details",
    "plans": "Policy_Details",
    "product_data": "Policy_Details",
    "plan_data": "Policy_Details",
}

COL_MAP = {
    "policy_number": "AppID",
    "policy_id": "AppID",
    "customer_id": "ClientID",
    "client_id": "ClientID",
    "sales_person_id": "SALES_ID",
    "servicing_agent_id": "SALES_ID",
    "agent_id": "SALES_ID",
    "claim_id": "Claim_No",
    "claim_number": "Claim_No",
    "product_name": "Plan.Name",
    "plan_name": "Plan.Name",
    "product_id": "Prod_No",
    "plan_id": "Prod_No",
    "sum_assured": "Sum.Assured",
    "annual_premium": "APE",
    "modal_premium": "APE",
    "policy_status": "POLICY.STATUS",
    "sales_channel": "CHANNEL",
    "channel": "CHANNEL",
    "location_city": "Owner.Comm.City.Name",
    "city": "Owner.Comm.City.Name",
    "state": "Owner.Comm.State",
    "age": "Owner.Age",
    "gender": "Owner.Gender",
    "annual_income": "Owner.Earned.Income",
    "earned_income": "Owner.Earned.Income",
    "occupation": "Owner.Occupation",
    "education": "Owner.Education",
    "claim_amount": "Claim.Amount",
    "claim_status": "Status.of.Claim",
    "cause_of_death": "Cause.of.Death",
    "persistency_bucket": "Persistency_Bucket",
    "renewal_status": "Renewal_Status",
    "renewal_premium_amount": "Renewal_Premium_Amount",
    "rcu_rejection_rate": "RCU Rejections Rate",
    "pivc_mismatch_rate": "PIVC Number Mismatch Rate",
    "pivc_concern_rate": "PIVC Concern Raised Rate",
    "free_look_rate": "Free Look  Cancellation Rate",
    "first_year_persistency": "First Year Persistency Rate",
}

def clean_table_list(tables):
    mapped = []
    for t in tables:
        t_clean = str(t).lower().strip()
        mapped_t = TABLE_MAP.get(t_clean, "Policy_Details")
        if mapped_t not in mapped:
            mapped.append(mapped_t)
    return mapped if mapped else ["Policy_Details"]

def clean_col_list(cols):
    mapped = []
    for c in cols:
        c_clean = str(c).lower().strip()
        mapped_c = COL_MAP.get(c_clean, c)
        if mapped_c not in mapped:
            mapped.append(mapped_c)
    return mapped

def update_legacy_ps():
    legacy_path = os.path.join(PS_DIR, "problem_statements.json")
    if not os.path.exists(legacy_path):
        return
    with open(legacy_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    updated = []
    for r in records:
        r_new = dict(r)
        r_new["relevant_tables"] = clean_table_list(r.get("relevant_tables", []))
        r_new["relevant_columns"] = clean_col_list(r.get("relevant_columns", []))
        
        # Replace sql template references
        sql = r_new.get("sql_template", "")
        if sql:
            for old_t, new_t in TABLE_MAP.items():
                sql = re.sub(rf'\b{old_t}\b', new_t, sql, flags=re.IGNORECASE)
            for old_c, new_c in COL_MAP.items():
                sql = re.sub(rf'\b{old_c}\b', f'"{new_c}"', sql, flags=re.IGNORECASE)
            r_new["sql_template"] = sql
            
        # Replace cypher template references
        cypher = r_new.get("cypher_template", "")
        if cypher:
            cypher = cypher.replace("Policy_data", "Policy").replace("Customer_data", "Owner").replace("Sales_Info", "Agent").replace("Claims_data", "Claim")
            r_new["cypher_template"] = cypher
            
        # Example queries
        eqs = []
        for eq in r.get("example_queries", []):
            eq_clean = re.sub(r'\bPOL\d+\b', '1000000015', eq, flags=re.IGNORECASE)
            eq_clean = re.sub(r'\bCUST\d+\b', '7140735859', eq_clean, flags=re.IGNORECASE)
            eq_clean = re.sub(r'\bAGT\d+\b', '270084', eq_clean, flags=re.IGNORECASE)
            eq_clean = re.sub(r'\bCLM\d+\b', '490', eq_clean, flags=re.IGNORECASE)
            eq_clean = eq_clean.replace("Term Life", "Canara HSBC Guaranteed Fortune Plan").replace("Whole Life", "Canara HSBC Wealth Edge")
            eqs.append(eq_clean)
        r_new["example_queries"] = eqs
        updated.append(r_new)

    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    print(f"Updated {len(updated)} records in {legacy_path}")

def update_business_ps():
    biz_path = os.path.join(PS_DIR, "business_problem_statements.json")
    if not os.path.exists(biz_path):
        return
    with open(biz_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    updated = []
    for r in records:
        r_new = dict(r)
        r_new["relevant_tables"] = clean_table_list(r.get("relevant_tables", []))
        r_new["relevant_columns"] = clean_col_list(r.get("relevant_columns", []))
        
        # Replace sample queries
        eqs = []
        for eq in r.get("example_queries", []):
            eq_clean = re.sub(r'\bPOL\d+\b', '1000000015', eq, flags=re.IGNORECASE)
            eq_clean = re.sub(r'\bCUST\d+\b', '7140735859', eq_clean, flags=re.IGNORECASE)
            eq_clean = re.sub(r'\bAGT\d+\b', '270084', eq_clean, flags=re.IGNORECASE)
            eq_clean = re.sub(r'\bCLM\d+\b', '490', eq_clean, flags=re.IGNORECASE)
            eq_clean = eq_clean.replace("Term Life", "Canara HSBC Guaranteed Fortune Plan").replace("Whole Life", "Canara HSBC Wealth Edge")
            eqs.append(eq_clean)
        r_new["example_queries"] = eqs
        updated.append(r_new)

    with open(biz_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    print(f"Updated {len(updated)} records in {biz_path}")

if __name__ == "__main__":
    update_legacy_ps()
    update_business_ps()
