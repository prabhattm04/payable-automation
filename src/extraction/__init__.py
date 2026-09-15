"""src/extraction — OCR and raw evidence extraction layers."""
from src.extraction.ocr import (
    OCRBlock,
    PageOCRResult,
    DocumentOCRResult,
    OCRProvider,
    PP_OCRv5Provider,
    RapidOCRProvider,
)
from src.extraction.candidates import (
    ChargeCandidate,
    DiscountCandidate,
    DocumentIdentityCandidate,
    ExtractionCandidates,
    LineCandidate,
    PartyIdentityCandidate,
    POCandidate,
    TaxCandidate,
    TotalCandidate,
    adapt_vision_evidence_to_candidates,
    extract_candidates_from_document,
    extract_candidates_from_page,
    normalize_currency_with_context,
)
from src.extraction.consolidation import (
    ConsolidatedField,
    FieldStatus,
    consolidate_candidates,
    consolidate_document_groups,
)
from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.extraction.validation import (
    ExtractionValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    validate_financial_document_assemblies,
    validate_financial_document_assembly,
)

__all__ = [
    "ChargeCandidate",
    "ConsolidatedField",
    "DiscountCandidate",
    "DocumentIdentityCandidate",
    "DocumentOCRResult",
    "ExtractionCandidates",
    "ExtractionValidationResult",
    "FieldStatus",
    "FinancialDocumentAssembly",
    "LineCandidate",
    "OCRBlock",
    "OCRProvider",
    "PP_OCRv5Provider",
    "PageOCRResult",
    "PartyIdentityCandidate",
    "POCandidate",
    "RapidOCRProvider",
    "TaxCandidate",
    "TotalCandidate",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationStatus",
    "adapt_vision_evidence_to_candidates",
    "assemble_financial_documents",
    "consolidate_candidates",
    "consolidate_document_groups",
    "extract_candidates_from_document",
    "extract_candidates_from_page",
    "normalize_currency_with_context",
    "validate_financial_document_assemblies",
    "validate_financial_document_assembly",
]


