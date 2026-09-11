# Data Dictionary: Talk to Data PoC (Metadata V4)

This dictionary documents the **5 production CSV tables** used exclusively across the **Third Draft** architecture for Canara HSBC Life Insurance analytics and retrieval.

---

## 1. Dataset Inventory

| File Name | Rows | Columns | Primary Grain | Role in Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **`Policy_Details.csv`** | 25,000 | 16 | 1 row per policy / application `AppID` | **Base / Fact Table** |
| **`Owner_Details.csv`** | 25,000 | 12 | 1 row per customer linked to `AppID` | **Lookup / Demographics** |
| **`Persistency_Details.csv`** | 24,542 | 12 | 1 row per renewal cohort per `AppID` | **Lookup / Renewal Performance** |
| **`Sales_Details.csv`** | 1,956 | 10 | 1 row per `SALES_ID` | **Lookup / Agent Compliance & Quality** |
| **`Claims_Details.csv`** | 134 | 9 | 1 row per claim linked to `AppID` | **Lookup / Mortality & Settlement Experience** |

---

## 2. Table Schemas & Field Definitions

### 2.1 `Policy_Details.csv`
*Core policy master and policy-level contractual attributes.*

| Field Name | Type | Description | Sample Values / Range |
| :--- | :--- | :--- | :--- |
| **`AppID`** | Identifier (PK) | Application/policy-level identifier used as the universal join key across Policy, Owner, Persistency and Claims. | `5000635514`, `9102471149`, `9102206801` |
| **`SALES_ID`** | Identifier (FK) | Sales user / advisor identifier linking to sales quality indicators. | `977337`, `970998`, `270084` |
| **`Branch_No`** | Identifier | Branch identifier where the policy was booked. | `83546`, `43702`, `30749` |
| **`Prod_No`** | Identifier | Product code associated with the policy plan. | `P22`, `P69`, `P94` |
| **`POLICY.STATUS`** | Categorical | Current policy servicing status. | `Premium paying (regular)`, `Lapsed`, `Surrendered`, `Death` |
| **`CHANNEL`** | Categorical | Sales distribution channel. | `Banca` |
| **`Login.Date`** | Date | Application login date. | `7/25/2023`, `8/29/2024` |
| **`Inforce.Date`** | Date | Date on which policy became active/in-force. | `7/29/2023`, `8/30/2024` |
| **`Inforce.month`** | Period Code | Period code for in-force month. | `20237`, `20248` |
| **`Trad.ULIP`** | Categorical | Product classification between Traditional and Unit-Linked (ULIP). | `TRAD`, `ULIP` |
| **`Plan.Name`** | Text | Full commercial plan name. | `Canara HSBC Life Insurance Guaranteed Fortune Plan`, `Wealth Edge` |
| **`Mode`** | Categorical | Premium payment mode/frequency. | `Annually` |
| **`PREMIUM.PAYING.TERM`** | Numeric | Premium paying term in years. | `7`, `10`, `12` |
| **`Policy.Term`** | Numeric | Total policy maturity term in years. | `12`, `15`, `18` |
| **`Sum.Assured`** | Numeric (INR) | Policy sum assured / death benefit coverage amount. | `142850.0`, `540347.0`, `1000000.0` |
| **`APE`** | Numeric (INR) | Annualized Premium Equivalent value. | `25000.0`, `50000.0`, `100000.0` |

---

### 2.2 `Owner_Details.csv`
*Customer demographic, geographic, and socioeconomic attributes.*

| Field Name | Type | Description | Sample Values / Range |
| :--- | :--- | :--- | :--- |
| **`AppID`** | Identifier (FK) | Policy application identifier linking to `Policy_Details`. | `1000000015`, `1000000033` |
| **`ClientID`** | Identifier (PK) | Customer/client unique identifier for the policy owner. | `7140735859`, `3775493405`, `7946770095` |
| **`Owner.Given.Name`** | Text | Given name of policyholder (treated as PII in reporting). | `RAVI KUMAR`, `GARGI`, `SHASHI` |
| **`Owner.Gender`** | Categorical | Gender of policy owner. | `M`, `F`, `C` |
| **`Owner.Age`** | Numeric | Age of the policy owner in years. | `40`, `38`, `64` |
| **`Owner.Education`** | Categorical | Education level. | `Graduate`, `Std XII pass`, `Post graduate & above` |
| **`Owner.Occupation`** | Categorical | Broad nature of occupation. | `Business Owner`, `Salaried`, `Agriculture`, `Self Employed` |
| **`Owner.Earned.Income`** | Numeric (INR) | Annual earned income of the owner. | `450000`, `1000000`, `1500000` |
| **`Owner.Comm.City.Name`** | Text | Communication city name. | `NORTH EAST DELHI`, `GHAZIABAD`, `BHIWANI` |
| **`Owner.Comm.State`** | Categorical | Communication state / region. | `Uttar Pradesh`, `Karnataka`, `Delhi`, `West Bengal` |
| **`Owner.Comm.Country`** | Categorical | Country of communication (NRI flag derived from non-India). | `India`, `United Arab Emirates`, `Bahrain` |
| **`Owner.Comm.Postal.Code`** | Identifier | PIN / Postal code. | `110094`, `201005`, `560001` |

---

### 2.3 `Persistency_Details.csv`
*Policy renewal transactions across multi-year cohort buckets.*

| Field Name | Type | Description | Sample Values / Range |
| :--- | :--- | :--- | :--- |
| **`AppID`** | Identifier (FK) | Policy application identifier linking to `Policy_Details`. | `1000010875`, `1350017290` |
| **`SALES_ID`** | Identifier (FK) | Sales user identifier for renewal servicing. | `977337`, `270084` |
| **`POLICY.STATUS`** | Categorical | Servicing status during renewal tracking. | `Premium paying (regular)`, `Lapsed` |
| **`CHANNEL`** | Categorical | Sales channel. | `Banca` |
| **`Inforce.Date`** | Date | Policy inception date. | `7/29/2023`, `8/30/2024` |
| **`PREMIUM.PAYING.TERM`**| Numeric | Premium paying term in years. | `7`, `10`, `12` |
| **`Renewal_Due_Date`** | Date | Date on which renewal premium is due. | `7/29/2026`, `8/30/2026` |
| **`Renewal_Paid_Date`** | Date | Date on which renewal premium was collected. | `8/4/2026`, `5/28/2026` |
| **`Renewal_Status`** | Categorical | Renewal collection status. | `Paid`, `Pending` |
| **`Renewal_Premium_Amount`** | Numeric (INR) | Premium payable for renewal cycle. | `25000`, `50000`, `100000` |
| **`Persistency_Bucket`** | Categorical | Measurement cohort period. | `13th Month`, `25th Month`, `37th Month` |
| **`Payment_Mode`** | Categorical | Premium collection mechanism. | `ECS/Auto-Debit`, `Online/UPI`, `Cheque`, `Cash`, `Credit Card` |

---

### 2.4 `Sales_Details.csv`
*Sales agent-level quality, compliance, and persistency KPIs.*

| Field Name | Type | Description | Sample Values / Range |
| :--- | :--- | :--- | :--- |
| **`SALES_ID`** | Identifier (PK) | Advisor / Sales Agent unique identifier. | `270084`, `800722`, `800812` |
| **`Surrender Rate`** | Metric (%) | Advisor-level policy surrender rate. | `0.0%` to `15.0%` |
| **`Early Claim Rate`** | Metric (%) | Mortality claims occurring within initial policy years. | `0.0%` to `5.0%` |
| **`PIVC Number Mismatch Rate`** | Metric (%) | Pre-Issuance Verification Call phone mismatch rate. | `0.0%` to `10.0%` |
| **`PIVC Concern Raised Rate`** | Metric (%) | Customer concerns flagged during verification calls. | `0.0%` to `12.0%` |
| **`RCU Rejections Rate`** | Metric (%) | Risk Control Unit fraud/risk rejections rate. | `0.0%` to `8.0%` |
| **`Free Look Cancellation Rate`** | Metric (%) | Policy cancellations during statutory 15/30 day window. | `0.0%` to `6.0%` |
| **`Sales Related Complaint`** | Metric (Count) | Number of advisor-related customer grievances. | `0`, `1`, `2`, `5` |
| **`First Year Persistency Rate`** | Metric (%) | Advisor 13th month persistency performance rate. | `60.0%` to `95.0%` |
| **`Type`** | Categorical | Agent status flag. | `Active`, `Inactive` |

---

### 2.5 `Claims_Details.csv`
*Policy death claims, intimation dates, causes of mortality, and settlement payouts.*

| Field Name | Type | Description | Sample Values / Range |
| :--- | :--- | :--- | :--- |
| **`AppID`** | Identifier (FK) | Policy application identifier linking to `Policy_Details`. | `1000010875`, `1350017290` |
| **`Claim_No`** | Identifier (PK) | Unique claim reference number. | `490`, `333`, `607` |
| **`Date.of.Death`** | Date | Date of death of the life assured. | `30-May-25`, `28-Nov-23` |
| **`Cause.of.Death`** | Categorical | Medical cause of mortality. | `SUDDEN DEATH`, `HEART ATTACK`, `NATURAL DEATH`, `ACCIDENT` |
| **`Claim.Amount`** | Numeric (INR) | Claim amount claimed by nominee. | `57750.0`, `1250000.0`, `1702500.0` |
| **`Date.of.Intimation`** | Date | Date insurer was notified of claim. | `6-Mar-26`, `18-Jun-25` |
| **`Status.of.Claim`** | Categorical | Current claim adjudication state. | `Paid`, `WIP` |
| **`Amount.Paid`** | Numeric (INR) | Total settlement amount disbursed to beneficiary. | `57750.0`, `1250000.0`, `1702500.0` |
| **`Settlement.Date`** | Date | Date claim disbursement completed. | `31-Mar-26`, `31-Jul-25` |
