"""Dump the first 500 chars of each text-readable PDF for spot-checking."""
import json

with open('analysis/document_inventory.json', encoding='utf-8') as f:
    data = json.load(f)

from src.pdf.text_extractor import extract_document_text
from pathlib import Path

text_readable = [d for d in data['documents'] if d['text'].get('text_coverage', 0) > 0]
print(f"Text-readable documents: {len(text_readable)}")
print()

for d in text_readable:
    doc = extract_document_text(Path('documents') / d['filename'])
    text = doc.full_text
    print(f"{'='*60}")
    print(f"FILE: {d['filename']}  ({d['page_count']} pages, {d['text']['total_characters']} chars)")
    print(f"{'='*60}")
    print(text[:1500])
    print()
