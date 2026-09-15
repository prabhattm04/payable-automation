"""run_pipeline.py — End-to-End Pipeline Orchestrator for the Bookable Payable Architecture.

Coordinates the complete pipeline from input PDF to final validated JSON output:
    PDF
     ↓
    PDF discovery
     ↓
    Evidence acquisition (cached OCR or native rendering + RapidOCR)
     ↓
    PageUnderstanding (classify_page)
     ↓
    DocumentGrouping (group_document)
     ↓
    Selective Qwen routing (route_page)
     ↓
    Candidate extraction (extract_candidates_from_document)
     ↓
    Candidate consolidation (consolidate_document_groups)
     ↓
    Financial assembly (assemble_financial_documents)
     ↓
    Extraction validation (validate_financial_document_assembly)
     ↓
    Master-data matching (Supplier, Buyer, PO, Tax, PaymentTerms)
     ↓
    FinancialStructure normalization (normalize_financial_structure)
     ↓
    ERP reconstruction (reconstruct_erp)
     ↓
    Reconciliation (reconcile)
     ↓
    9D Safety Gate (evaluate_payable_safety)
     ↓
    9E Payable Builder (build_payable / build_declined_entry)
     ↓
    File output (build_file_output / write_file_output)

Design Principles:
------------------
1. Thin Coordinator: Does NOT perform accounting formulas, tax calculations, or custom matching.
2. Interface Fidelity: Reuses existing Phase 5 -> 9E public entrypoints without alteration.
3. Master Matching: Strict source DTO construction; passes matches only to existing consumers.
4. Lazy Vision: Only initializes VisionProvider if a page is explicitly routed to Vision.
5. Failure Isolation: Exceptions in one PDF do not abort the batch; failed documents are
   counted separately as processing failures.
6. Deterministic & Isolated: Preserves multi-page and supporting document isolation (DU-02).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from src.utils.logging import configure_logging, get_logger

# ── PDF Layer ─────────────────────────────────────────────────────────────
from src.pdf.loader import PdfFile, discover_pdfs
from src.pdf.renderer import DEFAULT_DPI, render_pdf

# ── Understanding Layer ───────────────────────────────────────────────────
from src.understanding.evidence import (
    Evidence,
    PageEvidence,
    ocr_json_to_page_evidence,
    page_ocr_result_to_evidence,
)
from src.understanding.page_classifier import (
    PageRole,
    PageUnderstanding,
    PayableRelevance,
    classify_page,
)
from src.understanding.document_grouper import (
    DocumentGroup,
    GroupingResult,
    group_document,
)
from src.understanding.qwen_router import (
    RouterDecisionType,
    RoutingDecision,
    route_page,
)

# ── Vision Abstraction ────────────────────────────────────────────────────
from src.vision.provider import VisionProvider

# ── Extraction Layer ──────────────────────────────────────────────────────
from src.extraction.ocr import RapidOCRProvider, save_page_ocr_json
from src.extraction.candidates import (
    ExtractionCandidates,
    extract_candidates_from_document,
)
from src.extraction.consolidation import consolidate_document_groups
from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.extraction.validation import (
    ExtractionValidationResult,
    validate_financial_document_assembly,
)

# ── Master Data Layer ─────────────────────────────────────────────────────
from src.matching.store import MasterDataStore
from src.matching.supplier_matcher import SupplierMatcher
from src.matching.buyer_matcher import BuyerMatcher
from src.matching.po_matcher import POMatcher
from src.matching.tax_matcher import TaxMatcher
from src.matching.payment_terms_matcher import PaymentTermsMatcher
from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedBuyerIdentity,
    ObservedPaymentTermIdentity,
    ObservedPOIdentity,
    ObservedPOLineEvidence,
    ObservedSupplierIdentity,
    ObservedTaxIdentity,
)

# ── Accounting & Output Layer ─────────────────────────────────────────────
from src.accounting.financial_structure import (
    FinancialStructure,
    normalize_financial_structure,
)
from src.accounting.erp_reconstruction import (
    ERPReconstruction,
    reconstruct_erp,
)
from src.accounting.reconciliation import (
    ReconciliationResult,
    reconcile,
)
from src.accounting.decision import (
    PayableDecision,
    PayableDecisionStatus,
    evaluate_payable_safety,
)
from src.accounting.payable_builder import (
    PayableContractError,
    build_declined_entry,
    build_file_output,
    build_payable,
    validate_file_output,
    write_file_output,
)

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Master Matchers Registry Container
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MasterMatchers:
    """Holds shared master-data store and initialized matchers."""
    store: MasterDataStore
    supplier: SupplierMatcher
    buyer: BuyerMatcher
    po: POMatcher
    tax: TaxMatcher
    payment_terms: PaymentTermsMatcher

    @classmethod
    def load(cls, master_data_dir: Union[str, Path] = "master_data") -> MasterMatchers:
        p = Path(master_data_dir)
        store = MasterDataStore.from_directory(p)
        return cls(
            store=store,
            supplier=SupplierMatcher(store),
            buyer=BuyerMatcher(store),
            po=POMatcher(store),
            tax=TaxMatcher(store),
            payment_terms=PaymentTermsMatcher(store),
        )


# ══════════════════════════════════════════════════════════════════════════
# Execution Statistics and Result Tracking
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentProcessResult:
    """Detailed outcome for one processed document."""
    file_name: str
    success: bool
    page_count: int = 0
    group_count: int = 0
    vision_escalation_count: int = 0
    assembly_count: int = 0
    decisions: List[str] = field(default_factory=list)
    payables_count: int = 0
    declined_count: int = 0
    output_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class PipelineRunSummary:
    """Aggregated batch statistics."""
    total_documents: int = 0
    successful_executions: int = 0
    processing_failures: int = 0
    decisions: Dict[str, int] = field(default_factory=lambda: {
        PayableDecisionStatus.SAFE_TO_AUTODRAFT.value: 0,
        PayableDecisionStatus.HOLD_FOR_REVIEW.value: 0,
        PayableDecisionStatus.UNSAFE_TO_AUTODRAFT.value: 0,
        PayableDecisionStatus.NOT_PAYABLE.value: 0,
    })
    payables_emitted: int = 0
    declined_emitted: int = 0
    output_files_written: int = 0
    document_results: List[DocumentProcessResult] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print clean summary matching Section 21 specification."""
        print("\n" + "=" * 60)
        print("PIPELINE RUN SUMMARY")
        print("=" * 60)
        print(f"Processed: {self.total_documents}")
        print(f"Successful pipeline executions: {self.successful_executions}")
        print(f"Processing failures: {self.processing_failures}\n")
        print(f"SAFE_TO_AUTODRAFT: {self.decisions.get(PayableDecisionStatus.SAFE_TO_AUTODRAFT.value, 0)}")
        print(f"HOLD_FOR_REVIEW: {self.decisions.get(PayableDecisionStatus.HOLD_FOR_REVIEW.value, 0)}")
        print(f"UNSAFE_TO_AUTODRAFT: {self.decisions.get(PayableDecisionStatus.UNSAFE_TO_AUTODRAFT.value, 0)}")
        print(f"NOT_PAYABLE: {self.decisions.get(PayableDecisionStatus.NOT_PAYABLE.value, 0)}\n")
        print(f"Payables emitted: {self.payables_emitted}")
        print(f"Declined entries emitted: {self.declined_emitted}")
        print(f"Output files written: {self.output_files_written}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# Lazy Vision Provider Factory
# ══════════════════════════════════════════════════════════════════════════

class LazyVisionProviderHolder:
    """Conditionally instantiates PuterQwenProvider only when required."""
    def __init__(self) -> None:
        self._provider: Optional[VisionProvider] = None
        self._attempted: bool = False

    def get_provider(self) -> Optional[VisionProvider]:
        if not self._attempted:
            self._attempted = True
            try:
                from src.vision.puter_qwen import PuterQwenProvider
                self._provider = PuterQwenProvider()
            except Exception as e:
                log.warning("Vision provider initialization deferred or unavailable: %s", e)
                self._provider = None
        return self._provider


# ══════════════════════════════════════════════════════════════════════════
# Core Evidence Acquisition Helper
# ══════════════════════════════════════════════════════════════════════════

def _extract_page_sort_key(p: Path) -> int:
    stem = p.stem
    match = re.search(r"\d+", stem)
    return int(match.group(0)) if match else 0


def acquire_document_evidence(
    pdf_file: PdfFile,
    artifacts_dir: Path = Path("artifacts"),
) -> List[PageEvidence]:
    """Obtain PageEvidence records for a document via cached OCR or native rendering + RapidOCR.
    
    1. Looks for cached artifacts/ocr/<stem>/page_*.json.
    2. If missing, renders pages using render_pdf (DPI=200) and runs RapidOCR.
    3. Resolves and attaches valid image_path so downstream vision routing is viable.
    """
    stem = pdf_file.stem
    ocr_dir = artifacts_dir / "ocr" / stem

    page_evidences: List[PageEvidence] = []

    if ocr_dir.is_dir():
        cached_jsons = sorted(ocr_dir.glob("page_*.json"), key=_extract_page_sort_key)
        if cached_jsons:
            for json_path in cached_jsons:
                pe = ocr_json_to_page_evidence(json_path)
                page_evidences.append(pe)

    # Fallback to dynamic rendering & OCR if cache is missing or empty
    if not page_evidences:
        log.info("No OCR cache found for %s; rendering and running RapidOCR...", pdf_file.filename)
        rendered_dir = artifacts_dir / "rendered_pages"
        render_result = render_pdf(
            pdf_path=pdf_file.path,
            output_dir=rendered_dir,
            dpi=DEFAULT_DPI,
            stem=stem,
        )

        ocr_provider = RapidOCRProvider()
        for page_res in render_result.pages:
            if not page_res.success or not page_res.image_path:
                continue
            ocr_res = ocr_provider.ocr_page(
                image_path=page_res.image_path,
                document_id=pdf_file.filename,
                page_number=page_res.page_number,
            )
            save_page_ocr_json(ocr_res, artifacts_dir / "ocr")
            pe = page_ocr_result_to_evidence(ocr_res)
            page_evidences.append(pe)

    # Ensure valid image_path on every PageEvidence
    for pe in page_evidences:
        if pe.image_path and Path(pe.image_path).exists():
            continue
        candidate_img = artifacts_dir / "rendered_pages" / stem / f"page_{pe.page_number:03d}.png"
        if candidate_img.exists():
            pe.image_path = str(candidate_img)

    return page_evidences


# ══════════════════════════════════════════════════════════════════════════
# Document-Level Orchestrator
# ══════════════════════════════════════════════════════════════════════════

def process_document(
    pdf_file: PdfFile,
    matchers: MasterMatchers,
    output_dir: Path,
    artifacts_dir: Path = Path("artifacts"),
    vision_provider_getter: Optional[Callable[[], Optional[VisionProvider]]] = None,
    disable_vision: bool = False,
    strict: bool = False,
) -> DocumentProcessResult:
    """Execute the full pipeline on a single PDF document.
    
    Preserves document boundary, isolates supporting pages, avoids formula execution,
    evaluates safety gate, and produces exactly one compliant output file.
    """
    file_name = pdf_file.filename
    try:
        # 1. Evidence Acquisition
        page_evidences = acquire_document_evidence(pdf_file, artifacts_dir=artifacts_dir)
        if not page_evidences:
            raise RuntimeError(f"Could not acquire page evidence for {file_name}")

        page_count = len(page_evidences)

        # 2. Page Classification (7B)
        understandings = [classify_page(pe) for pe in page_evidences]

        # 3. Document Grouping (7C)
        grouping_result = group_document(page_evidences, understandings)
        groups = grouping_result.groups
        group_count = len(groups)

        # 4. Selective Vision Routing (7D)
        routing_decisions: List[RoutingDecision] = []
        vision_required_count = 0
        for pe, und in zip(page_evidences, understandings):
            grp = next((g for g in groups if pe.page_number in g.page_numbers), None)
            r_dec = route_page(page_evidence=pe, page_understanding=und, group=grp)
            routing_decisions.append(r_dec)
            if r_dec.decision == RouterDecisionType.ROUTE_TO_VISION:
                vision_required_count += 1

        # 5. Conditional Vision Provider Resolution
        active_vision_provider: Optional[VisionProvider] = None
        if not disable_vision and vision_required_count > 0 and vision_provider_getter is not None:
            active_vision_provider = vision_provider_getter()

        # 6. Candidate Extraction (9B-1)
        candidates = extract_candidates_from_document(
            page_evidences=page_evidences,
            understandings=understandings,
            groups=groups,
            routing_decisions=routing_decisions,
            vision_provider=active_vision_provider,
        )

        # 7. Candidate Consolidation (9B-2)
        facts_list = consolidate_document_groups(candidates)

        # 8. Financial Assembly (9B-3)
        assemblies = assemble_financial_documents(facts_list, groups)
        assembly_count = len(assemblies)

        payables: List[Dict[str, Any]] = []
        declined: List[Dict[str, Any]] = []
        decisions: List[str] = []

        # 9. Process Each Logical Assembly Independently
        for asm in assemblies:
            # A. Extraction Validation (9B-4)
            val_result = validate_financial_document_assembly(asm)

            # B. Financial Structure Normalization (9C-1)
            struct = normalize_financial_structure(asm, val_result)

            # C. Master Data Matching (8B, 8C, 8D) — Strictly Typed DTOs
            # 1. Supplier
            obs_supplier = ObservedSupplierIdentity(
                name=asm.supplier.observed_name,
                vat_id=asm.supplier.vat_id,
                country=asm.supplier.country,
                email=asm.supplier.email,
                bank_iban=asm.supplier.bank_iban,
                address=asm.supplier.address,
                evidence_ids=list(asm.supplier.evidence_ids),
                field_evidence_ids={k: v[0] for k, v in asm.supplier.field_evidence_ids.items() if v},
            )
            supplier_match = matchers.supplier.match(obs_supplier)

            # 2. Buyer
            obs_buyer = ObservedBuyerIdentity(
                company_name=asm.buyer.observed_company,
                company_code=asm.buyer.company_code,
                business_unit_name=asm.buyer.business_unit,
                business_unit_code=asm.buyer.business_unit_code,
                location_name=asm.buyer.location,
                location_code=asm.buyer.location_code,
                invoice_to_address=asm.buyer.invoice_to_address,
                evidence_ids=list(asm.buyer.evidence_ids),
                field_evidence_ids={k: v[0] for k, v in asm.buyer.field_evidence_ids.items() if v},
            )
            buyer_match = matchers.buyer.match(obs_buyer)

            # 3. Purchase Order
            po_lines: List[ObservedPOLineEvidence] = []
            for ln in asm.lines:
                po_lines.append(
                    ObservedPOLineEvidence(
                        description=ln.description,
                        quantity=ln.quantity,
                        unit_price=ln.unit_price,
                        amount=ln.amount,
                        uom=ln.metadata.get("uom") or ln.raw_values.get("uom"),
                        evidence_id=ln.evidence_ids[0] if ln.evidence_ids else None,
                    )
                )
            obs_po = ObservedPOIdentity(
                po_number=asm.po.observed_po_number,
                supplier_id=supplier_match.master_id if (supplier_match and supplier_match.status == MatchStatus.MATCHED) else None,
                supplier_name=asm.supplier.observed_name,
                currency=asm.currency,
                gross_amount=asm.printed_totals.gross_total if asm.printed_totals else None,
                lines=po_lines,
                evidence_ids=list(asm.po.evidence_ids),
                field_evidence_ids={k: v[0] for k, v in asm.po.field_evidence_ids.items() if v},
            )
            po_match = matchers.po.match(obs_po)

            # 4. Payment Term
            pt_raw = asm.facts.identity.raw_values.get("payment_terms") or asm.facts.identity.metadata.get("payment_terms")
            obs_term = ObservedPaymentTermIdentity(
                raw_text=pt_raw,
                evidence_ids=list(asm.facts.identity.evidence_ids),
            )
            payment_term_match = matchers.payment_terms.match(obs_term)

            # 5. Tax (Header taxes matched against tax master)
            tax_matches: List[MasterMatchResult] = []
            for tx in asm.taxes:
                obs_tx = ObservedTaxIdentity(
                    tax_name=tx.tax_name,
                    tax_type=tx.tax_type,
                    rate=tx.rate,
                    country=asm.supplier.country if asm.supplier else None,
                    placement="header",
                    evidence_ids=list(tx.evidence_ids),
                )
                tax_matches.append(matchers.tax.match(obs_tx))

            # D. ERP Reconstruction (9C-2)
            recon = reconstruct_erp(struct)

            # E. Reconciliation (9C-3)
            recon_result = reconcile(recon, tolerance=Decimal("0.00"))

            # F. 9D Safety Gate (Contract: supplier, buyer, po only)
            decision = evaluate_payable_safety(
                reconciliation=recon_result,
                reconstruction=recon,
                document_facts=asm.facts,
                supplier_match=supplier_match,
                buyer_match=buyer_match,
                po_match=po_match,
            )
            decisions.append(decision.status.value)

            # G. 9E Output Generation
            if decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT:
                payable = build_payable(
                    decision=decision,
                    financial_structure=struct,
                    reconstruction=recon,
                    supplier_match=supplier_match,
                    buyer_match=buyer_match,
                    po_match=po_match,
                    payment_term_match=payment_term_match,
                )
                payables.append(payable)
            elif decision.status == PayableDecisionStatus.NOT_PAYABLE:
                declined_entry = build_declined_entry(decision)
                declined.append(declined_entry)
            else:
                # HOLD_FOR_REVIEW / UNSAFE_TO_AUTODRAFT emit neither payable nor declined
                pass

        # 10. Construct & Write File Output Envelope
        file_payload = build_file_output(
            file_name=file_name,
            payables=payables,
            declined=declined,
            strict=strict,
        )
        written_path = write_file_output(
            file_payload=file_payload,
            output_dir=output_dir,
            strict=strict,
        )

        return DocumentProcessResult(
            file_name=file_name,
            success=True,
            page_count=page_count,
            group_count=group_count,
            vision_escalation_count=vision_required_count,
            assembly_count=assembly_count,
            decisions=decisions,
            payables_count=len(payables),
            declined_count=len(declined),
            output_path=written_path,
        )

    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        log.error("Error processing %s: %s\n%s", file_name, err_msg, traceback.format_exc())

        # Safe fallback: write valid empty envelope to satisfy 1-output-per-input contract
        fallback_path: Optional[Path] = None
        try:
            fallback_payload = build_file_output(
                file_name=file_name,
                payables=[],
                declined=[],
                strict=False,
            )
            fallback_path = write_file_output(
                file_payload=fallback_payload,
                output_dir=output_dir,
                strict=False,
            )
        except Exception as write_err:
            log.error("Failed writing fallback envelope for %s: %s", file_name, write_err)

        return DocumentProcessResult(
            file_name=file_name,
            success=False,
            output_path=fallback_path,
            error=err_msg,
        )


# ══════════════════════════════════════════════════════════════════════════
# Batch Pipeline Runner
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    documents_dir: Union[str, Path] = "documents",
    output_dir: Union[str, Path] = "output",
    master_data_dir: Union[str, Path] = "master_data",
    artifacts_dir: Union[str, Path] = "artifacts",
    filter_files: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    disable_vision: bool = False,
    strict: bool = False,
) -> PipelineRunSummary:
    """Run the end-to-end pipeline across discovered PDFs."""
    doc_path = Path(documents_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Discovering PDFs in %s...", doc_path)
    all_pdfs = discover_pdfs(doc_path, recursive=True)

    # Deterministic sorting is already guaranteed by discover_pdfs
    if filter_files:
        normalized_filters = {Path(f).name.lower() for f in filter_files}
        all_pdfs = [p for p in all_pdfs if p.filename.lower() in normalized_filters or p.stem.lower() in normalized_filters]

    if limit is not None and limit > 0:
        all_pdfs = all_pdfs[:limit]

    total_docs = len(all_pdfs)
    log.info("Total PDF(s) to process: %d", total_docs)

    # Master Data matchers initialization (loaded once)
    log.info("Loading master data store from %s...", master_data_dir)
    matchers = MasterMatchers.load(master_data_dir)

    # Lazy Vision Provider holder (deferred instantiation)
    lazy_vision = LazyVisionProviderHolder()

    summary = PipelineRunSummary(total_documents=total_docs)

    for idx, pdf_file in enumerate(all_pdfs, start=1):
        log.info("[%d/%d] Starting %s", idx, total_docs, pdf_file.filename)
        res = process_document(
            pdf_file=pdf_file,
            matchers=matchers,
            output_dir=out_path,
            artifacts_dir=Path(artifacts_dir),
            vision_provider_getter=lazy_vision.get_provider,
            disable_vision=disable_vision,
            strict=strict,
        )
        summary.document_results.append(res)

        if res.success:
            summary.successful_executions += 1
            for d in res.decisions:
                summary.decisions[d] = summary.decisions.get(d, 0) + 1
            summary.payables_emitted += res.payables_count
            summary.declined_emitted += res.declined_count
        else:
            summary.processing_failures += 1

        if res.output_path is not None and res.output_path.exists():
            summary.output_files_written += 1

        # Format concise progress output per Section 21
        decisions_str = ", ".join(res.decisions) if res.decisions else ("FAILED" if not res.success else "NONE")
        print(f"[{idx}/{total_docs}] {pdf_file.filename}")
        print(f"  pages: {res.page_count}")
        print(f"  groups: {res.group_count}")
        print(f"  vision: {res.vision_escalation_count}")
        print(f"  assemblies: {res.assembly_count}")
        print(f"  decisions: {decisions_str}")
        print(f"  payables: {res.payables_count}")
        if not res.success:
            print(f"  ERROR: {res.error}")

    summary.print_summary()
    return summary


# ══════════════════════════════════════════════════════════════════════════
# CLI Entrypoint
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bookable Payable End-to-End Pipeline Orchestrator"
    )
    parser.add_argument(
        "--documents", "--documents-dir",
        dest="documents",
        type=Path,
        default=Path("documents"),
        help="Root directory containing input PDFs (default: documents)",
    )
    parser.add_argument(
        "--output", "--output-dir",
        dest="output",
        type=Path,
        default=Path("output"),
        help="Destination directory for JSON autodraft envelopes (default: output)",
    )
    parser.add_argument(
        "--master-data", "--master-data-dir",
        dest="master_data",
        type=Path,
        default=Path("master_data"),
        help="Directory containing master data JSONs (default: master_data)",
    )
    parser.add_argument(
        "--artifacts", "--artifacts-dir",
        dest="artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing OCR and rendered page artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Subset of PDF filenames to process (e.g. --files INV-01.pdf INV-02.pdf)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to process",
    )
    parser.add_argument(
        "--no-vision",
        dest="no_vision",
        action="store_true",
        help="Disable vision escalation even if routed to vision",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict schema and semantic validation during envelope writing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging output",
    )

    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level)

    summary = run_pipeline(
        documents_dir=args.documents,
        output_dir=args.output,
        master_data_dir=args.master_data,
        artifacts_dir=args.artifacts,
        filter_files=args.files,
        limit=args.limit,
        disable_vision=args.no_vision,
        strict=args.strict,
    )

    if summary.processing_failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
