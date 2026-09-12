# FMN Inventory Attention Engine

**FMN AI Engineer Internship Technical Assessment — Project 1: Supply Chain**

> **Know which SKUs need attention before they become a problem, and understand why.**

The FMN Inventory Attention Engine is a decision support prototype built for the Supply Chain problem described in the FMN AI Engineer Internship Technical Assessment.

It helps a Supply Chain planner identify which SKUs require attention, understand the underlying inventory risk, and investigate the reasons behind each flag.

The system combines demand forecasting, inventory risk assessment, prioritisation, plain language AI explanations, and grounded question answering.

---

## 1. Problem Understanding

The sponsor's problem is:

> "We keep getting caught off guard — some SKUs run out and delay production, others sit overstocked and tie up working capital. I don't have anything today that tells me, ahead of time, which SKUs need attention and why."

I interpreted this as a **decision support problem**, not simply a forecasting problem.

A forecast by itself does not tell a planner what requires action. The tool therefore connects:

- Expected demand
- Current inventory
- Lead time
- Expected replenishment
- Projected inventory position
- Potential stockout
- Potential excess inventory
- Data quality and uncertainty

The product question is therefore:

> **Which SKUs need attention, how urgent are they, and why?**

The system is designed to help the planner make that decision rather than automatically making replenishment decisions on their behalf.

---

## 2. Solution

The **FMN Inventory Attention Engine** transforms the supplied daily supply chain data into a prioritised attention queue.

At a high level:

```text
Raw Supply Chain Data
        |
        v
Data Preparation
        |
        v
Demand Forecast
        |
        v
Inventory Risk Engine
        |
        v
Attention Prioritisation
        |
        +-------------------+
        |                   |
        v                   v
   SKU Analysis        Validation
        |
        v
Structured Risk Drivers
        |
        v
Grounded LLM
        |
        +-------------------+
        |                   |
        v                   v
AI Explanation       Ask the Data