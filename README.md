# Autonomous Bookable Payable Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/Tests-732%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/Architecture-Deterministic%20Pipeline-orange.svg)]()

An end-to-end, production-grade document processing system that transforms complex, multilingual supplier PDF documents into structured, bookable **autodraft** records that an ERP accounting system can book without human intervention.

---

## 1. Executive Summary & The Mandate

Given a supplier PDF document, this system determines:
1. **What the document is** (Invoice, Credit Memo, Delivery Note, Order Confirmation, or Non-payable supporting material).
2. **What, if anything, is owed**, extracting line items, charges, discounts, and multi-tier tax treatments.
3. **Master Data Resolution**: Links observed supplier entities, buyer organizational hierarchy (Company, Business Unit, Location), tax classifications, purchase orders, and payment terms against master records.
4. **Accounting Integrity**: Satisfies the exact mathematical and structural constraints of the sealed ERP oracle (`erp.py`), guaranteeing that every emitted autodraft reconstructs the exact gross amount owed down to the cent without fabricated or artificial balancing figures.

### The Core Problem: Visual Copying vs. Accounting Truth
A document extractor that faithfully copies visual text blocks will fail in real-world ERP systems. Accounting documents contain:
- Mixed currencies across invoice pages and supporting timesheets/receipts.
- Withholding taxes, cascading levies, and compound rates stated at varying line vs. header levels.
- Non-payable attachments (bank confirmations, delivery receipts, terms of service) bundled with invoices.
- Ambiguous or partial supplier identities requiring robust fuzzy and relational resolution.

This pipeline does not merely extract tokens — it reconstructs the underlying **financial fact model**, normalizes its accounting structure, reconciles arithmetic across three independent sources of truth, and enforces rigorous safety gates before clearing any autodraft.

---

## 2. Pipeline Architecture

The pipeline processes documents through a multi-stage sequential architecture:

```
                          ┌────────────────────────┐
                          │    Supplier PDF File   │
                          └───────────┬────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      ┌──────────────────────┐                  ┌──────────────────────┐
      │  PyMuPDF Text Extract│                  │  PyMuPDF DPI-200 PNG │
      └──────────┬───────────┘                  └──────────┬───────────┘
                 │                                         │
                 ▼                                         ▼
      ┌──────────────────────┐                  ┌──────────────────────┐
      │ Text Heuristics      │                  │ RapidOCR (PP-OCRv5)  │
      │ (Fast Classification)│                  │ (Latin / Thai ONNX)  │
      └──────────┬───────────┘                  └──────────┬───────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Unified Evidence Model    │
                        │ (BBoxes, Text, Confidence)│
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Page Role Classifier      │
                        │ & Document Grouper        │
                        └─────────────┬─────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               │ [Confidence Gate]                           │
               ▼                                             ▼
     [High OCR Confidence]                         [Low Confidence / Scan]
               │                                             │
               │                                             ▼
               │                                   ┌───────────────────┐
               │                                   │ Qwen-VL via Puter │
               │                                   └─────────┬─────────┘
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Semantic Candidate Extract│
                        │ (Dates, Parties, Lines,   │
                        │  Taxes, Gross, Subtotals) │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Fact Model Consolidation  │
                        │ (Immutable DocumentFacts) │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Master Data Matchers      │
                        │ - Supplier (Tax ID/Fuzzy) │
                        │ - Buyer Org Hierarchy     │
                        │ - Tax Master (Rate/Type)  │
                        │ - PO & Payment Terms      │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Financial Assembly        │
                        │ & Extraction Validation   │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Structure Normalization   │
                        │ & ERP Reconstruction      │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ 3-Way Reconciliation      │
                        │ & Safety Decision Arbiter │
                        └─────────────┬─────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
           [SAFE_TO_AUTODRAFT]               [HOLD_FOR_REVIEW / UNSAFE]
                     │                                 │
                     ▼                                 ▼
        ┌─────────────────────────┐       ┌─────────────────────────┐
        │ Emit Bookable Autodraft │       │ Route to Declined / Hold│
        │ in output/<id>.json     │       │ with Detailed Reasons   │
        └─────────────────────────┘       └─────────────────────────┘
```

---

## 3. Project Structure

```
├── erp.py                     # Sealed ERP Oracle (ground-truth recompute)
├── example_check.py           # Verification script for testing autodraft against erp.py
├── run_pipeline.py            # Primary CLI orchestrator for batch and single-file runs
├── AUTODRAFT_SCHEMA.md        # Canonical autodraft JSON schema specification
├── README.md                  # Project overview, setup, and architecture documentation
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template for vision inference
├── .gitignore                 # Standard Python/project ignore rules
│
├── master_data/               # Master reference datasets (JSON)
│   ├── suppliers.json         # Known vendor identities, VAT/tax IDs, addresses
│   ├── chart_of_books.json    # Organizational hierarchy (Company, BU, Location)
│   ├── tax_master.json        # Tax types, jurisdiction codes, rates
│   ├── payment_terms.json     # Standard payment terms and identifiers
│   └── po_master.json         # Purchase order references and line mappings
│
├── documents/                 # Challenge evaluation PDF corpus (36 documents)
├── models/                    # Offline ONNX models & dictionaries for OCR
│   └── ocr/
│       ├── latin/             # PP-OCRv5 Latin recognition model & dictionary
│       └── thai/              # PP-OCRv5 Thai recognition model & dictionary
│
├── src/                       # Production source code
│   ├── accounting/            # Financial structure normalization, ERP reconstruction, reconciliation, safety decision
│   ├── extraction/            # OCR engine, candidate extraction, fact consolidation, financial assembly, validation
│   ├── matching/              # Deterministic & fuzzy master-data matchers (Supplier, Buyer, Tax, PO, Terms)
│   ├── understanding/         # Unified evidence models, page classification, document grouping, vision router
│   ├── pdf/                   # High-fidelity PyMuPDF rendering & native text extraction
│   ├── inventory/             # Document inventory and feature classification
│   ├── vision/                # Puter AI / Qwen-VL vision API provider
│   └── utils/                 # Logging and string normalization utilities
│
├── tests/                     # Comprehensive test suite (28 test modules, 730+ tests)
└── analysis/                  # Research benchmarks, hardware reports, and inventory manifests
```

---

## 4. Setup & Installation

### Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/prabhattm04/payable-automation.git
cd payable-automation
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment (Optional for Vision Escalation)
The pipeline runs completely offline using local PyMuPDF and RapidOCR with bundled ONNX models. If you wish to enable the optional Qwen Vision escalation for low-confidence scans:
```bash
cp .env.example .env
# Edit .env and insert your Puter API key:
# PUTER_API_KEY=your_token_here
```

---

## 5. Usage & CLI Commands

### Single Documented Command (Processes all `documents/` into `output/`)
In accordance with the project contract, the system runs with a single command over the input document folder and generates `output/<stem>.json` for each file:

```bash
python run_pipeline.py --documents-dir documents --output-dir output
```

### Single Document Execution
To execute the pipeline on an individual document:
```bash
python run_pipeline.py --filter INV-01.pdf --output-dir output
```

### Validating Outputs with the ERP Oracle
To test any generated autodraft JSON against the sealed ERP oracle (`erp.py`):
```bash
# Using the example checker:
python example_check.py output/INV-01.json

# Or directly invoking erp.py:
python erp.py output/INV-01.json
```

Example output:
```text
will_book_gross = 438.0 EUR
```

---

## 6. Engineering & Implementation Highlights

### 1. Dual-Tier Text Extraction (RapidOCR + PP-OCRv5 ONNX)
- Native PyMuPDF text extraction extracts character vectors and font metadata when available.
- For rendered page images and scanned documents, the pipeline employs `RapidOCR` backed by ONNX Runtime.
- Includes language-aware recognition dictionaries (`latin` and `thai`) to accurately transcribe diacritics and complex non-Latin scripts (e.g. Thai tax forms in `DU-05`).

### 2. Multi-Page Document Grouping & Classification
- Analyzes layout headers, continuation footers, page numbering patterns ("Page 1 of 2"), and semantic signals.
- Splits bundled PDFs into logical units (e.g., isolating invoice front pages from non-payable timesheets, packing slips, or vendor terms).
- Identifies and routes supporting documents to `declined[]` with explicit audit reasons rather than polluting the accounting ledger.

### 3. High-Precision Master Data Matching
Master reference resolution is isolated into specialized matchers in `src/matching/`:
- **Supplier Matching**: Priority cascade: Tax ID/VAT lookup -> exact normalized name -> alias matching -> token-set ratio fuzzy matching against `master_data/suppliers.json`.
- **Buyer Organization Resolution**: Hierarchical resolution matching the company name, business unit, and facility address against `master_data/chart_of_books.json`.
- **Tax Classification**: Matches stated document tax rates and tax names (e.g., standard VAT, MwSt, NHIL, GETFund, COVID levy) against `master_data/tax_master.json`.
- **Purchase Order & Terms Matching**: Extracts and verifies PO identifiers against `master_data/po_master.json` and parses terms like "Net 30" into standard payment term codes.

### 4. 3-Way Reconciliation & Safety Gates
Before emitting an autodraft as `SAFE_TO_AUTODRAFT`, the decision engine (`src/accounting/decision.py`) evaluates three concurrent mathematical views:
1. **Document Stated Total**: What the invoice explicitly declares as owed.
2. **Component Recomputation**: The sum of line items + taxes - discounts + freight/levies.
3. **ERP Oracle Simulation**: The exact gross that `erp.py` computes from the structured payload.

```text
┌────────────────────────────────────────────────────────┐
│                   Decision Statuses                    │
├──────────────────────┬─────────────────────────────────┤
│ SAFE_TO_AUTODRAFT    │ 3-way reconciliation matches to │
│                      │ the cent. Emitted to payables[].│
├──────────────────────┼─────────────────────────────────┤
│ HOLD_FOR_REVIEW      │ Arithmetic matches but master   │
│                      │ code is unverified or minor     │
│                      │ field is ambiguous.             │
├──────────────────────┼─────────────────────────────────┤
│ UNSAFE_TO_AUTODRAFT  │ Arithmetic mismatch, multi-cur- │
│                      │ rency conflict, or non-payable. │
│                      │ Routed to declined[].           │
└──────────────────────┴─────────────────────────────────┘
```

---

## 7. Testing & Quality Assurance

The codebase includes an extensive suite of unit, integration, and regression tests.

```bash
# Run the complete test suite
pytest
```

### Test Suite Summary
- **Total Test Files**: 28 modules in `tests/`
- **Total Tests**: 738 tests (732 passing)
- **Coverage Areas**:
  - `test_pdf/`: Rendering fidelity and text extraction.
  - `test_ocr.py`: RapidOCR ONNX provider and polygon bounding boxes.
  - `test_master_data.py`: Supplier, buyer, tax, PO, and payment term matching algorithms.
  - `test_document_grouper.py`: Multi-page bundle segmentation and role tagging.
  - `test_candidate_extraction.py`: Pattern parsing across diverse invoice layouts.
  - `test_reconciliation.py`: 3-way discrepancy checks and tolerance evaluation.
  - `test_payable_decision.py`: End-to-end safety arbiter decisions on real corpus documents.

---

## 8. Design Reflections & The Three Foundational Questions

### 1. What did you eventually understand about these documents that you did not understand on day one?

On day one, the natural inclination is to view this challenge as a classical OCR and layout extraction problem: *find the invoice header, extract the table rows, read the tax and total, match strings against master data, and dump JSON*. That mental model stalls quickly because accounting systems do not evaluate strings; they execute a computational contract.

The core realization is that **visual layout is an unreliable proxy for accounting structure**:

1. **The Fallacy of Visual Copying vs. Computational Atomicity**:
   The grading oracle `erp.py` is sealed. It takes no pre-calculated gross or line tax totals; it recomputes the entire gross from raw constituent atoms:
   $$\text{Gross} = \sum \Big( (\text{qty} \times \text{unit\_price} - \text{discount}) \times (1 + \text{line\_tax}) \Big) + \text{header\_taxes} + \text{charges} - \text{withholdings}$$
   Many real invoices print numbers that are already net of line discounts, or print line items that already embed tax, or present freight and handling as separate tabular rows. If an extraction engine copies what the eye sees into `quantity`, `unit_price`, and `tax_amount`, the ERP recalculates taxes on top of already-taxed amounts, causing massive footing discrepancies. You cannot extract the printed total; you must extract the **atomic formula inputs** that compel the ERP to arrive at that exact total legitimately.

2. **Tax Placement as a Legal Semantics Contract**:
   Two structural configurations can foot to the exact same cent total, yet only one is legally bookable. For instance, extracting three invoice lines that each carry a 19% VAT rate and collapsing them into a single header-level tax produces the identical gross, but violates accounting line-item allocation rules and fails ERP validation. Conversely, statutory cascading levies—such as Ghana's NHIL (2.5%), GETFund (2.5%), and COVID-19 Health Levy (1%) on `INV-19`—are computed on subtotal, and standard VAT (15%) is applied on the subtotal plus levies. Placing those levies on individual lines corrupts the tax base. Tax placement (header vs. line) dictates *accounting jurisdiction and liability*, not just algebra.

3. **Documents as Evidentiary Packets, Not 1:1 Invoices**:
   In enterprise accounts payable, a PDF is rarely a clean, single-page invoice. It is an evidentiary bundle containing an invoice cover sheet, timesheets, delivery receipts, customs declarations, and terms of service. On day one, one assumes every page belongs to the payable. In reality, page 5 of `DU-02` is a packing list with HTS commodity codes (`3822190080`) that naive regexes mistake for unit prices, and pages 2–20 are Turkish Lira expense receipts. The problem was never "extract the invoice"; the problem was first "discover the boundaries of the payable obligation within an evidentiary bundle".

---

### 2. When your system meets a document unlike any it has seen, what does it actually *do* — and why does that generalise instead of guessing?

When an unseen document enters the pipeline, the system avoids overfitting to known layouts through a four-stage deterministic grounding architecture:

1. **Geometric Coordinate Anchoring (No Rigid Layout Templates)**:
   The system uses zero hardcoded document templates. Every word, block, and number is ingested as an `Evidence` object anchored to its 2D spatial coordinates `[x0, y0, x1, y1]`, polygon geometry, page number, and confidence score. Tabular candidate extraction (`src/extraction/candidates.py`) relies on spatial geometry:
   - Column headers are discovered by dynamic keyword clustering across horizontal bands.
   - Column intervals are projected vertically downward through the page body.
   - Row boundaries are formed by spatial clustering of bounding boxes along the vertical axis.
   Because table extraction is governed by 2D geometry rather than vendor-specific regexes, a layout never seen before is parsed using the same spatial invariants as standard invoices.

2. **Decoupled 3-Way Reconciliation Arbiters**:
   Rather than trusting a single extraction path, the engine builds three concurrent, decoupled models:
   - **Path A (Observed Surface Model)**: What the vendor explicitly printed (e.g. printed subtotal, printed tax amount, printed gross).
   - **Path B (Decomposed Component Assembly)**: The bottom-up sum of discovered line items, calculated discounts, line taxes, and extra charges.
   - **Path C (ERP Oracle Simulation)**: The simulated output of feeding the assembled payload into `erp.py`.
   The reconciliation engine (`src/accounting/reconciliation.py`) computes deltas across all three paths. Discrepancies are categorized into specific failure modes (e.g., `UNRESOLVED_TAX_VARIANCE`, `DISCOUNT_LEVY_COLLAPSE`, `LINE_SUM_MISMATCH`).

3. **Principled Refusal as a Primary Operational Mode**:
   The system strictly enforces Rule 1 (*"Never invent a figure to balance the books"*) and Rule 2 (*"Every code must be a real match"*):
   - If a line item provides a lump-sum amount of €1,500 with no quantity or unit price, `quantity` remains `None`—it is never fabricated as `1`.
   - If a supplier's tax ID or name does not achieve high-confidence matching against `master_data/suppliers.json`, the master code remains `""` (empty string) rather than guessing a close neighbor.
   - If the 3-way reconciliation shows an arithmetic delta exceeding 0.01 currency units, the safety arbiter (`src/accounting/decision.py`) transitions the document to `HOLD_FOR_REVIEW` or `UNSAFE_TO_AUTODRAFT`, routing it to `declined[]` with an auditable explanation.

In real-world finance, an automation system that flags an ambiguous document for human review is valuable and safe; a system that invents numbers to force an automated balance is disastrous.

---

### 3. Was there a document you concluded could NOT be solved the way the others were? If so, which, and how did you know?

Yes. Two documents in the evaluation corpus proved that a universal, uniform extraction path will fail when a document asks something of the system that the page does not contain the answer to:

#### Primary Case: `DU-02.pdf` (The Multi-Currency, Mixed-Entity Composite Packet)
- **What the document demands**:
  `DU-02.pdf` is a dense 20-page document. Page 1 presents a consolidated commercial reimbursement invoice billing for **€4,800.00 EUR**. Pages 2 through 20 consist of Turkish hotel folios, taxi slips, restaurant chits, and customs packing lists denominated in **Turkish Lira (TRY)**.
- **Why it cannot be solved like the others**:
  Every standard invoice in the dataset is solved by extracting its line items, calculating their taxes, and footing them to the total. If you apply that universal procedure to `DU-02`:
  1. The line items on pages 2–20 are denominated in `TRY`, whereas the primary invoice is in `EUR`.
  2. The exchange rates, currency conversion dates, and banking margins used to roll up the Turkish Lira chits into €4,800.00 EUR are **literally not present on the pages**.
  3. No OCR engine or arithmetic routine can sum 45 Turkish Lira meal items and arrive at €4,800.00 without an external FX oracle that is not part of the input.
  The page *does not contain the data* required to reconcile line-item Turkish Lira expenses to the final consolidated Euro payable.
- **How we knew**:
  During pipeline execution, the reconciliation engine threw an arithmetic discrepancy exceeding 50,000 units because it was aggregating TRY numbers into a EUR payable. Simultaneously, the currency extractor detected conflicting ISO currency symbols across pages.
- **The Principled Solution**:
  Rather than fabricating conversion rates or forcing 20 pages of receipts into a single broken invoice, our `document_grouper` and `page_classifier` identify that `DU-02` contains a single primary payable (`Page 1`, EUR commercial invoice) followed by 19 pages of non-payable **supporting expense vouchers**. The pipeline:
  - Isolates Page 1 as the actionable payable.
  - Classifies Pages 2–20 as `SUPPORTING_ATTACHMENT`.
  - Emits the supporting vouchers into `decision.supporting_group_ids` for auditing, while declining them from `payables[]`.
  - Flags the overall document as `HOLD_FOR_REVIEW` because the cover invoice relies on un-itemized aggregate line summaries that cannot be cross-footed against attached receipts without external currency conversion tables.

#### Secondary Case: `HLD-01.pdf` (The 543-Year Temporal Calendar Shift)
- **The Anomaly**: `HLD-01` is a Thai service invoice where dates are stated in the **Buddhist Era (BE) calendar** (e.g., year `2569` BE).
- **The Problem**: Naive date parsers extract the year literally as `2569-02-15`. When passed to downstream ERP validation engines with calendar sanity checks, the document fails as an invalid date 543 years in the future.
- **The Solution**: The pipeline applies regional calendar normalization: recognizing Thai tax invoice context and transforming Buddhist Era dates to Gregorian CE years (`2569 - 543 = 2026 CE`) before generating the autodraft.

---

## 9. Submission Details

- **Repository**: [https://github.com/prabhattm04/payable-automation.git](https://github.com/prabhattm04/payable-automation.git)
- **Author**: Prabhat
- **Submission Date**: September 2026
