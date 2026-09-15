# Bookable Payable Pipeline — Full Implementation Handoff Document

> **Produced by:** Antigravity IDE (code-only agent)
> **Purpose:** Complete codebase audit for ChatGPT session handoff
> **Date verified:** 2026-09-14 (all findings from live source inspection)
> **Test baseline:** `434 passed, 0 failed` (`python -m pytest tests/ -q --tb=no`)

---

## 1. What the Project Is

An end-to-end **bookable payable processing pipeline** that:

1. Accepts physical PDF documents (invoices, credit memos, supporting attachments)
2. Extracts OCR text + renders page images
3. Classifies pages and groups multi-page logical documents
4. Selectively routes low-confidence pages to Qwen Vision (via Puter AI)
5. Extracts typed semantic candidates (invoice number, parties, PO, lines, taxes, totals)
6. Consolidates candidates into an immutable fact model (no arithmetic, no master matching)
7. Matches observed facts against master data (suppliers, buyer hierarchy, tax codes, POs, payment terms)
8. Produces an `AUTODRAFT` JSON record validated against a sealed ERP oracle (`erp.py`)

**Key constraint**: The grading oracle `erp.py` is immutable and defines "correctness". All output must supply raw components (not pre-calculated totals) and `erp.py` recomputes gross totals.

---

## 2. Repository Root

```
d:\candidate_kit\
├── erp.py                     <- Sealed grading oracle (DO NOT MODIFY)
├── README.md                  <- Mandate, rules, correctness definition
├── AUTODRAFT_SCHEMA.md        <- Output record structure definition
├── .env.example               <- Requires PUTER_API_KEY
├── requirements.txt
├── master_data/               <- Static reference data (JSON)
│   ├── suppliers.json
│   ├── chart_of_books.json
│   ├── tax_master.json
│   ├── payment_terms.json
│   └── po_master.json
├── pdfs/                      <- Input PDF corpus (immutable test fixtures)
├── src/                       <- All implementation code
│   ├── pdf/
│   ├── extraction/
│   ├── understanding/
│   ├── matching/
│   ├── inventory/
│   ├── vision/
│   └── utils/
└── tests/                     <- pytest test suite
```

---

## 3. Phase Architecture Map

| Phase | Directory / File | Description |
|-------|-----------------|-------------|
| PDF Rendering | `src/pdf/renderer.py` | PDF to PNG page images (DPI configurable) |
| PDF Text | `src/pdf/text_extractor.py` | Native PDF text via PyMuPDF |
| Inventory | `src/inventory/classifier.py` | Heuristic document classification (signals only) |
| OCR | `src/extraction/ocr.py` | RapidOCR / PP_OCRv5 text extraction |
| Phase 7A | `src/understanding/evidence.py` | Unified Evidence model + provenance |
| Phase 7B | `src/understanding/page_classifier.py` | Deterministic page role + payable relevance |
| Phase 7C | `src/understanding/document_grouper.py` | Multi-page logical document grouping |
| Phase 7D | `src/understanding/qwen_router.py` | Selective Qwen Vision escalation routing |
| Phase 8A | `src/matching/` (5 files) | Master data loading, indexing, normalization |
| Phase 8B | `src/matching/supplier_matcher.py` | Supplier master-data matching |
| Phase 8B | `src/matching/buyer_matcher.py` | Buyer hierarchical matching |
| Phase 8C | `src/matching/tax_matcher.py` | Tax code master matching |
| Phase 8D | `src/matching/po_matcher.py` | PO matching |
| Phase 8D | `src/matching/payment_terms_matcher.py` | Payment terms matching |
| Phase 9A | `src/understanding/document_facts.py` | Immutable typed fact model |
| Phase 9B-1 | `src/extraction/candidates.py` | Semantic candidate extraction |
| Phase 9B-2 | `src/extraction/consolidation.py` | Candidate to DocumentFacts consolidation |
| Vision | `src/vision/provider.py` + `puter_qwen.py` | Qwen Vision provider (abstract + concrete) |

---

## 4. Data Flow (Sequential Pipeline)

```
PDF File
  |
  |-> src/pdf/renderer.py         -> PNG images per page
  |-> src/pdf/text_extractor.py   -> PageText (native text, char counts)
  |
  |-> src/inventory/classifier.py -> Inventory signals (doc type heuristics)
  |
  |-> src/extraction/ocr.py       -> OCR text blocks with bboxes + confidence
  |
  |-> src/understanding/
  |     evidence.py               -> Phase 7A: PageEvidence (Evidence items with evidence_ids)
  |     page_classifier.py        -> Phase 7B: PageUnderstanding (role, payable_relevance)
  |     document_grouper.py       -> Phase 7C: GroupingResult (logical DocumentGroup partitions)
  |     qwen_router.py            -> Phase 7D: RoutingDecision (OCR sufficient vs. escalate)
  |
  |-> src/vision/puter_qwen.py    -> (if routed) Qwen Vision response -> Evidence
  |
  |-> src/extraction/candidates.py (Phase 9B-1)
  |     -> ExtractionCandidates per page (all typed candidate lists)
  |
  |-> src/extraction/consolidation.py (Phase 9B-2)
  |     -> DocumentFacts (consolidated, immutable, evidence-traced)
  |
  |-> src/matching/ (Phase 8A-8D)
        -> MasterMatchResult for supplier, buyer, tax, PO, payment_terms
        -> AUTODRAFT JSON record (fed to erp.py for grading)
```

---

## 5. Implemented Files — Detailed Inventory

### 5.1 `src/pdf/renderer.py`
- **Status:** COMPLETE
- **Exports:** `render_pdf_pages(pdf_path, output_dir, dpi=150) -> list[Path]`
- **Behaviour:** Renders each page as PNG; graceful page-level exception isolation
- **Dependency:** PyMuPDF (`fitz`)

### 5.2 `src/pdf/text_extractor.py`
- **Status:** COMPLETE
- **Exports:** `extract_document_text(pdf_path) -> DocumentText`
- **Models:** `PageText` (page_number, raw_text, char_count, word_count, native_text_available, fonts, has_images, width_pt, height_pt), `DocumentText`
- **Threshold:** `IMAGE_ONLY_THRESHOLD = 20` non-whitespace chars marks page as image-only
- **Design:** Never performs OCR

### 5.3 `src/inventory/classifier.py`
- **Status:** COMPLETE
- **Purpose:** Heuristic inventory signals from native PDF text only (not final extraction)
- **Design:** No filename rules, no master matching, no final autodraft fields
- **Patterns:** Currency codes, invoice/credit/debit/tax/freight words, all regex-based, multilingual

### 5.4 `src/extraction/ocr.py`
- **Status:** COMPLETE
- **OCR Providers:** `RapidOCR` (primary), `PP_OCRv5` (secondary/configurable)
- **Feature:** Page-level exception isolation
- **Output:** Text blocks with bounding boxes and per-block confidence scores

### 5.5 `src/understanding/evidence.py` — Phase 7A
- **Status:** COMPLETE (~764 lines)
- **Core types:**
  - `EvidenceSource` enum: `ocr | vision | native_pdf | manual | derived`
  - `EvidenceProvenance`: source, document_id, page_number, extraction_method, confidence
  - `Evidence`: content, bbox, polygon, confidence, evidence_id (deterministic SHA-256)
  - `PageEvidence`: page_number, document_id, items, image_path
- **Key function:** `generate_evidence_id(...)` — SHA-256 hash of (document_id, page_number, source, method, content, bbox, polygon, index). Never uses timestamps or file paths.
- **Key adapter:** `vision_response_to_evidence()` — converts Qwen VisionResponse to Evidence
- **Design:** Conflict coexistence — OCR vs Vision observations coexist as separate Evidence items

### 5.6 `src/understanding/page_classifier.py` — Phase 7B
- **Status:** COMPLETE
- **Enums:** `PageRole` (invoice, credit_memo, debit_memo, continuation, supporting, unknown), `PayableRelevance` (payable_candidate, supporting, ambiguous, non_payable)
- **Model:** `PageUnderstanding` (page_role, payable_relevance, signals dict, evidence_ids)
- **Entry point:** `classify_page(page_evidence: PageEvidence) -> PageUnderstanding`
- **Design:** 100% deterministic regex-based. Zero LLM calls. Zero filename rules. Multilingual.

### 5.7 `src/understanding/document_grouper.py` — Phase 7C
- **Status:** COMPLETE (~514 lines)
- **Models:** `DocumentGroup` (group_id, document_id, page_numbers, grouping_signals, confidence=None), `GroupingResult`
- **Entry point:** `group_document(pages, understandings) -> GroupingResult`
- **Algorithm:** Score-based joining. Shared document number (+10), sequential pagination (+15), continuation role (+8). Join requires score >= 8. Strong contradictions prevent joining.
- **Design:** Non-adjacent interleaved documents correctly supported. `confidence = None` always.

### 5.8 `src/understanding/qwen_router.py` — Phase 7D
- **Status:** COMPLETE (~414 lines)
- **Enums:** `RouterDecisionType` (ocr_sufficient, route_to_vision)
- **Model:** `RoutingDecision` (document_id, page_number, decision, reasons, signals, evidence_ids, confidence=None)
- **Entry point:** `route_page(page_evidence, page_understanding, group) -> RoutingDecision`
- **Routing triggers (any one causes ROUTE_TO_VISION):**
  1. Low OCR quality: mean confidence < 0.80 OR >30% blocks < 0.70
  2. Classification ambiguous: PageRole.UNKNOWN or PayableRelevance.AMBIGUOUS
  3. Complex layout: >=80 blocks AND >=4 columns AND >=5 table rows (or >=100 blocks AND >=3 columns)
  4. Complex financial: >=3 distinct adjustment types, or withholding+discounts, or discounts+>=2 tax+charges
  5. Grouping ambiguous: continuation role but isolated in 1-page group
- **Helper:** `execute_vision_escalation(decision, page_evidence, provider, ...) -> Optional[Evidence]`

### 5.9 `src/understanding/document_facts.py` — Phase 9A
- **Status:** COMPLETE (~1341 lines)
- **Enums:** `FactOrigin` (observed, derived, matched), `InvoiceType`, `SemanticRole`, `Placement`
- **Evidence contract:** `_validate_evidence_contract()` raises ValueError if any accounting field has a value without provenance:
  - OBSERVED: requires evidence_ids or field_evidence_ids
  - DERIVED: requires derivation_rule or derivation_source_fields
  - MATCHED: requires matched_result or match provenance
- **Key dataclasses (all frozen=True):**
  - `DocumentIdentityFacts`: invoice_number, invoice_date, due_date, invoice_type, currency
  - `SupplierIdentityFact`: observed_name, vat_id, country, email, bank_iban, address
  - `BuyerIdentityFact`: observed_company, business_unit, location, company_code, business_unit_code, location_code, invoice_to_address
  - `PartyIdentityFacts`: supplier + buyer container
  - `POFacts`: observed_po_number
  - `LineFact`: line_number, description, quantity, unit_price, amount, discount, taxes, semantic_role — NO ARITHMETIC EVER
  - `TaxFact`: tax_type, tax_name, rate, amount, placement
  - `DiscountFact`: name, rate, amount, placement
  - `ChargeFact`: name, rate, amount, placement
  - `PrintedTotalsFact`: subtotal, net, taxable_base, tax_total, gross_total, amount_due, payment_total
  - `FinancialFacts`: lines, discounts, charges, taxes, printed_totals, currency
  - `DocumentFacts`: top-level model with all sub-facts + conflicting_facts + provenance
- **Utility:** `to_decimal(val)` — multi-locale numeric strings to Python Decimal without float loss
- **All models:** support to_dict(), from_dict(), to_json(), from_json() round-trip serialization

### 5.10 `src/extraction/candidates.py` — Phase 9B-1
- **Status:** COMPLETE (~1450 lines)
- **Candidate types (all frozen=True):**
  - `DocumentIdentityCandidate`: field_name, raw_value, normalized_value, evidence_ids, source, page_number
  - `PartyIdentityCandidate`: party_role (supplier/buyer), field_name, raw_value
  - `POCandidate`: po_number reference
  - `LineCandidate`: source_row_number, description, qty, unit_price, amount, discount, raw_row, raw_cells, semantic_role
  - `TaxCandidate`: tax_name, tax_type, rate, amount, placement
  - `DiscountCandidate`: label, rate, amount, placement
  - `ChargeCandidate`: label, rate, amount, placement
  - `TotalCandidate`: total_type, raw_label, raw_value, normalized_value
  - `ExtractionCandidates`: container for all candidate types per page
- **Extraction functions:**
  - `extract_identity_from_evidence(page_evidence, ...) -> List[DocumentIdentityCandidate]`
  - `extract_party_and_po_from_evidence(page_evidence, ...) -> (party_list, po_list)`
  - `extract_totals_and_taxes_from_evidence(page_evidence, ...) -> (totals, taxes, discounts, charges)`
  - `extract_lines_from_evidence(page_evidence, ...) -> List[LineCandidate]` — bbox-based row clustering with 12px Y-tolerance
- **Line extraction:** Groups bboxes into rows, assigns qty/price/amount strictly by observed position. Never computes missing values.
- **Vision adapter:** `adapt_vision_evidence_to_candidates(vision_evidence, page_number, ...) -> (ident, party, po, lines, totals)`
- **Main entry points:**
  - `extract_candidates_from_page(page_evidence, understanding, group, routing_decision, vision_evidence, vision_provider) -> ExtractionCandidates`
  - `extract_candidates_from_document(page_evidences, ...) -> List[ExtractionCandidates]`
- **Currency normalization:** `normalize_currency_with_context(raw_curr, context_tokens)` — ISO mapping only when contextual evidence justifies it

### 5.11 `src/extraction/consolidation.py` — Phase 9B-2
- **Status:** COMPLETE (~1254 lines)
- **Enum:** `FieldStatus` (confirmed, complementary, ambiguous, conflicted, missing)
- **Consolidation functions (genuine conflicts never silently resolved):**
  - `consolidate_identity(candidates) -> (DocumentIdentityFacts, conflicts)`
  - `consolidate_supplier(candidates) -> (SupplierIdentityFact, conflicts)`
  - `consolidate_buyer(candidates) -> (BuyerIdentityFact, conflicts)`
  - `consolidate_po(candidates) -> (POFacts, conflicts)`
  - `consolidate_lines(candidates) -> (List[LineFact], conflicts)` — repeated identical lines from same source remain separate
  - `consolidate_taxes(candidates) -> (List[TaxFact], conflicts)` — complementary partials merged if compatible
  - `consolidate_discounts_and_charges(disc_cands, chg_cands) -> (discs, charges, conflicts)`
  - `consolidate_printed_totals(candidates) -> (PrintedTotalsFact, conflicts)`
- **Main entry points:**
  - `consolidate_candidates(candidates: List[ExtractionCandidates], payable_group_id, supporting_group_ids) -> DocumentFacts`
  - `consolidate_document_groups(candidates) -> List[DocumentFacts]`
- **Invariants:**
  - Candidate sort order is for determinism only, never precedence
  - Supporting document groups remain isolated
  - Date normalization: ISO and unambiguous DD.MM.YYYY only; ambiguous dates preserved as-is
  - Invoice type: header-placement candidates preferred over body mentions

### 5.12 `src/matching/` — Phase 8A through 8D
- **Status:** COMPLETE (all matchers + infrastructure)

**Phase 8A — Master Data Infrastructure:**

| File | Purpose |
|------|---------|
| `models.py` | SupplierRecord, CompanyRecord, BusinessUnitRecord, LocationRecord, TaxRecord, PaymentTermRecord, POLineRecord, PurchaseOrderRecord (all frozen dataclasses) |
| `loaders.py` | load_suppliers(), load_chart_of_books(), load_tax_master(), load_payment_terms(), load_po_master() |
| `normalization.py` | normalize_name(), normalize_identifier(), normalize_code(), normalize_iban(), normalize_currency(), normalize_rate(), normalize_text(), normalize_ascii_folding() |
| `indexes.py` | SupplierIndexes, ChartOfBooksIndexes, TaxIndexes, PaymentTermIndexes, PurchaseOrderIndexes — O(1) hash-map lookups |
| `store.py` | MasterDataStore — loads all masters once, builds all indexes via rebuild_indexes() |

**Phase 8B — Supplier and Buyer Matchers:**
- `match_models.py`: MatchStatus (matched/ambiguous/no_match — no "best_guess"), MasterMatchResult (confidence=None always), ObservedSupplierIdentity, ObservedBuyerIdentity, ObservedTaxIdentity, ObservedPOIdentity, ObservedPaymentTermIdentity
- `supplier_matcher.py`: SupplierMatcher.match(observed) -> MasterMatchResult
  - Priority: (1) VAT ID exact, (2) name+country, (3) email, (4) IBAN, (5) composite multi-signal
  - Cross-field conflicts -> AMBIGUOUS, not matched
- `buyer_matcher.py`: BuyerMatcher.match(observed) -> MasterMatchResult
  - Hierarchical: Company -> Business Unit -> Location
  - Cross-hierarchy conflict -> AMBIGUOUS

**Phase 8C — Tax Matcher:**
- `tax_matcher.py`: TaxMatcher.match(observed) -> MasterMatchResult
  - Matches by (country, rate) with +/-0.01% tolerance
  - Preserves tax_type_code for ERP output

**Phase 8D — PO and Payment Terms:**
- `po_matcher.py`: POMatcher.match(observed) -> MasterMatchResult
- `payment_terms_matcher.py`: PaymentTermsMatcher.match(observed) -> MasterMatchResult

### 5.13 `src/vision/`
- `provider.py`: VisionProvider (ABC): provider_name, model_name, complete_text(), analyze_image(); VisionResponse (success, content, model, provider, latency_seconds, error, metadata)
- `puter_qwen.py`: PuterQwenProvider(VisionProvider):
  - Endpoint: `https://api.puter.com/drivers/call`
  - Model: `qwen/qwen3-vl-plus-2025-12-19`
  - Auth: PUTER_API_KEY from env or .env file
  - Encodes image as base64 data URL in OpenAI-format multimodal message
  - API key redaction in all error messages
  - Alternate naming fallback (page-001 vs page_001)

---

## 6. Master Data Reference Files

### `master_data/suppliers.json`
Keys per entry: `supplier_id`, `name`, `vat_id`, `country`, `email`, `address`, `bank_iban`
~20+ supplier records. Unmatched supplier_id is a valid empty result.

### `master_data/chart_of_books.json`
Hierarchical: Company -> Business Unit -> Location
Keys: `company_code`, `company_name`, `business_units[].business_unit_code`, `locations[].location_code`, `invoice_to_address`

### `master_data/tax_master.json`
Keys: `code`, `country`, `tax_type`, `rate`, `name`

### `master_data/payment_terms.json`
Keys: `payment_term_id`, `days`, `text_aliases[]`

### `master_data/po_master.json`
Keys: `po_id`, `po_number`, `supplier_id`, `currency`, `po_lines[].line_id`, `description`, `quantity`, `uom`, `unit_price`

---

## 7. Grading Oracle (`erp.py`)

- **Status:** Sealed — immutable, stdlib-only, Python 3.10+
- **Accepts:** AUTODRAFT JSON record
- **Computes:** Gross total from raw components (lines, taxes, discounts, charges)
- **Contract:** Supply raw components; erp.py validates the arithmetic
- **NEVER:** Pre-calculate gross_total — let erp.py compute it

---

## 8. AUTODRAFT Schema (Output Format)

Defined in `AUTODRAFT_SCHEMA.md`. Key top-level fields:
- `invoice_number`, `invoice_date`, `due_date`, `invoice_type` (invoice/credit_memo/debit_memo/unknown)
- `currency`
- `supplier_id` (from Phase 8B match; empty string if no match)
- `buyer_company_code`, `buyer_business_unit_code`, `buyer_location_code`
- `po_number`, `payment_term_id`, `tax_type_code`
- `lines[]`: raw `{description, quantity, unit_price, amount}` — no pre-calculation
- `discounts[]`, `charges[]`: with `amount` and/or `rate`
- `taxes[]`: with `tax_type_code`, `rate`, `amount`

---

## 9. Test Suite

**Location:** `tests/`
**Baseline:** 434 passed, 0 failed, 5 deprecation warnings (SwigPy OCR ONNX libs — benign)

| Test File | Coverage |
|-----------|---------|
| `test_document_facts.py` | 18 scenarios: Phase 9A model serialization, evidence contracts, Decimal preservation, conflict coexistence, no-arithmetic, no-filename-classification |
| `test_candidate_extraction.py` | Phase 9B-1 extraction (identity, dates, invoice type, party, PO, lines, totals, taxes, vision adapter boundary) |
| `test_candidate_consolidation.py` | Phase 9B-2: consensus merging, conflict preservation, repeated-line safety, no master-IDs injection, multi-page grouping, determinism, real-corpus scenarios |
| `test_buyer_matcher.py` | Phase 8B buyer matching: hierarchical resolution, ambiguity, no-match, evidence preservation |
| `test_inventory.py` | Document inventory (PDF discovery, classifier signals) |
| Other files | OCR, grouper, router, supplier/tax/PO matchers |

---

## 10. Key Architectural Invariants (Non-Negotiable)

1. **Zero Arithmetic** — Never compute qty x price; never fill in missing amounts
2. **No Naked Accounting Fields** — Every value must have evidence_ids; `_validate_evidence_contract()` enforces this at construction time
3. **No Filename Classification** — Document type never inferred from filename or document_id
4. **No Fabricated Confidence** — All matchers and routers set `confidence = None`
5. **No Silent Conflict Resolution** — Conflicts go into `conflicting_facts`; no silent winner selection
6. **Repeated Lines Preserved** — Two identical physical rows remain two separate LineFact items
7. **Supporting Document Isolation** — Supporting docs' financial figures never aggregate into payable totals
8. **Deterministic Evidence IDs** — SHA-256 based; never timestamp- or path-dependent
9. **Oracle Boundary** — erp.py recomputes gross; never pass pre-computed ERP totals

---

## 11. Environment and Dependencies

### Required Environment Variable
```
PUTER_API_KEY=your_puter_api_key_here
```

### Key Runtime Dependencies
- `pymupdf` (fitz) — PDF rendering and native text extraction
- `rapidocr-onnxruntime` — Primary OCR engine
- `requests` — Puter API HTTP calls
- `python-dotenv` — .env file loading
- `pytest` — Test runner
- `decimal` (stdlib) — Exact numeric representation

### Python Version
Python 3.10+ (required for match statement and newer type annotations in erp.py)

---

## 12. Phase Implementation Completeness Summary

| Phase | File(s) | Status |
|-------|---------|--------|
| PDF Rendering | `src/pdf/renderer.py` | COMPLETE |
| Native Text | `src/pdf/text_extractor.py` | COMPLETE |
| Inventory | `src/inventory/classifier.py` | COMPLETE |
| OCR | `src/extraction/ocr.py` | COMPLETE |
| Phase 7A Evidence | `src/understanding/evidence.py` | COMPLETE |
| Phase 7B Classification | `src/understanding/page_classifier.py` | COMPLETE |
| Phase 7C Grouping | `src/understanding/document_grouper.py` | COMPLETE |
| Phase 7D Routing | `src/understanding/qwen_router.py` | COMPLETE |
| Phase 8A Master Data | `src/matching/` (5 files) | COMPLETE |
| Phase 8B Supplier/Buyer Match | `supplier_matcher.py`, `buyer_matcher.py` | COMPLETE |
| Phase 8C Tax Match | `tax_matcher.py` | COMPLETE |
| Phase 8D PO/PayTerms Match | `po_matcher.py`, `payment_terms_matcher.py` | COMPLETE |
| Phase 9A Fact Model | `src/understanding/document_facts.py` | COMPLETE |
| Phase 9B-1 Candidates | `src/extraction/candidates.py` | COMPLETE |
| Phase 9B-2 Consolidation | `src/extraction/consolidation.py` | COMPLETE |
| Vision Provider | `src/vision/provider.py` + `puter_qwen.py` | COMPLETE |
| AUTODRAFT Generation | not yet identified in src/ | NOT YET IMPLEMENTED |
| End-to-end pipeline runner | no main.py or pipeline.py found | NOT YET IMPLEMENTED |

---

## 13. Open Items / Next Implementation Steps

These are inferences from what IS implemented. The next session should verify and clarify before proceeding.

1. **AUTODRAFT Assembly** — No file found that takes `DocumentFacts` + `MasterMatchResult` objects and serializes them into the final AUTODRAFT JSON record. This is the final output assembly step:
   - Map DocumentFacts.identity.invoice_number -> AUTODRAFT.invoice_number
   - Map matched supplier_id from SupplierMatcher -> AUTODRAFT.supplier_id
   - Map buyer hierarchy codes from BuyerMatcher -> AUTODRAFT.buyer_*
   - Pass raw lines/taxes/discounts/charges as-is (no pre-calculation)
   - Feed result to erp.py for validation

2. **End-to-end Pipeline Runner** — No `main.py`, `pipeline.py`, or `run.py` found. Pipeline stages are individually implemented but not wired together into a single callable flow.

3. **Phase 9B-3 (if planned)** — Phase numbering goes to 9B-2. Check whether a Phase 9B-3 (matched fact enrichment from Phase 8 results) is planned.

4. **Vision Evidence End-to-End** — Verify Vision -> Evidence -> Candidates -> Facts works end-to-end with a real Puter API key.

5. **Multi-document Batch Orchestration** — No batch layer found; each PDF is processed individually.

---

## 14. Critical Reminders for Next Session

> [!IMPORTANT]
> The grading oracle `erp.py` is **sealed and immutable**. Never modify it. Never pre-compute gross totals; always supply raw line components.

> [!WARNING]
> `confidence = None` is a hard contract across all matchers and routers. Do not add fabricated confidence scores.

> [!IMPORTANT]
> All tests must continue to pass (434 passed). Any new implementation must add tests. Do not regress existing tests.

> [!NOTE]
> The `to_decimal()` function in `document_facts.py` handles multi-locale numeric strings. Use it for all monetary value normalization — do not use `float()`.
