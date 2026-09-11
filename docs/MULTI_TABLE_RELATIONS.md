# Multi-Table Relational Model: 5 Production Datasets

This document specifies the join relationships, integrity constraints, and query paths across the **5 Production Datasets** for Canara HSBC Life Insurance.

---

## 1. Relational Topology & Join Graph

```
                   +-----------------------+
                   |    Sales_Details      |
                   | (SALES_ID PK: 1,956)  |
                   +-----------+-----------+
                               |
                               | (SALES_ID FK)
                               v
+-------------------+      +---+-------------------+      +-----------------------+
|   Owner_Details   |<-----+  Policy_Details (Base)|----->| Persistency_Details   |
| (AppID, ClientID) |(AppID|  (AppID PK: 25,000)   |(AppID| (AppID PK: 24,542)    |
+-------------------+      +-----------+-----------+      +-----------------------+
                                       |
                                       | (AppID FK)
                                       v
                           +-----------+-----------+
                           |    Claims_Details     |
                           |  (Claim_No, AppID:134)|
                           +-----------------------+
```

---

## 2. Join Relationships & Granularity

| Primary Base Table | Related Table | Join Key | Relationship Cardinality | Business Usage |
| :--- | :--- | :--- | :--- | :--- |
| **`Policy_Details`** | **`Owner_Details`** | `AppID` | Many-to-One (or One-to-One per app) | Connects policy sales volume, term, and status with customer demographics (Age, Income, State, Gender). |
| **`Policy_Details`** | **`Sales_Details`** | `SALES_ID` | Many-to-One | Connects policy distribution channel performance with agent compliance flags (RCU, PIVC mismatch, grievances). |
| **`Policy_Details`** | **`Persistency_Details`** | `AppID` | **One-to-One** (24,542 of 25,000 policies) | Renewal collection status. Each policy carries **one** renewal record in **one** bucket (13M, 25M or 37M). |
| **`Policy_Details`** | **`Claims_Details`** | `AppID` | One-to-Many (subset) | Left join to evaluate claim severity, cause of death, and settlement turnaround against product plans. |

---

## 3. Standard SQL Join Patterns

### Multi-Dimensional Fact View
```sql
SELECT
    p."AppID",
    p."Plan.Name",
    p."POLICY.STATUS",
    p."Sum.Assured",
    p."APE",
    o."Owner.Given.Name",
    o."Owner.Age",
    o."Owner.Comm.State",
    s."First Year Persistency Rate",
    s."RCU Rejections Rate",
    c."Claim.Amount",
    c."Status.of.Claim",
    pers."Persistency_Bucket",
    pers."Renewal_Status"
FROM "Policy_Details" p
LEFT JOIN "Owner_Details" o ON p."AppID" = o."AppID"
LEFT JOIN "Sales_Details" s ON p."SALES_ID" = s."SALES_ID"
LEFT JOIN "Claims_Details" c ON p."AppID" = c."AppID"
LEFT JOIN "Persistency_Details" pers ON p."AppID" = pers."AppID";
```
