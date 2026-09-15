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

## 8. Design Reflections & Problem-Solving Approach

### 1. What did we understand about these documents that wasn't obvious on day one?
On day one, the challenge looks like an OCR and field-extraction task. It is not. The real problem is **structural accounting reconstruction**:
- A document can state a line total that includes tax, while the ERP expects net lines with separate tax objects.
- Placing a tax at the header when the document charged it per-line produces the same arithmetic total, but violates the ERP's placement semantics.
- Documents are often multi-currency bundles (e.g. `DU-02` contains a primary EUR invoice attached to TRY expense receipts). Treating the bundle as a single document corrupts the payable.

### 2. When the system encounters an unseen document, what does it do?
The system relies on **layered deterministic grounding**:
1. It never guesses or hallucinates numbers to force a balance. If an item cannot be grounded in the text, it is omitted.
2. If evidence is ambiguous, the safety arbiter flags the record as `HOLD_FOR_REVIEW` or `UNSAFE_TO_AUTODRAFT` rather than submitting an ungrounded draft.
3. Master data matching uses conservative confidence thresholds; when no entry matches confidently, the field remains empty as permitted by the schema.

### 3. Documents requiring specialized handling
- **`DU-02` (Multi-Currency Bundle)**: Contains an overarching EUR summary followed by individual TRY supporting vouchers. The document grouper and currency boundary detector isolate the primary payable from supporting evidence.
- **`DU-05` (Multilingual Thai Invoice)**: Standard Latin OCR fails on Thai script. Integrating PP-OCRv5 Thai ONNX models alongside language routing enables clean text acquisition without external cloud dependencies.
- **`INV-19` (Compound Levies & Service Quantity)**: Contains compound auxiliary taxes (NHIL, GETFund, COVID Levy) and flat service fees where unit price equals total amount. Handled by structured tax-line normalization preserving placement and quantity defaults.

---

## 9. Submission Details

- **Repository**: [https://github.com/prabhattm04/payable-automation.git](https://github.com/prabhattm04/payable-automation.git)
- **Author**: Prabhat
- **Submission Date**: September 2026
