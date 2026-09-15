import json

with open('analysis/document_inventory.json', encoding='utf-8') as f:
    data = json.load(f)

summary = data['summary']
docs = data['documents']

print('=== DATASET SUMMARY ===')
print('Total documents:', summary['total_documents'])
print('Total pages:', summary['total_pages'])
print('Single-page docs:', summary['single_page_documents'])
print('Multi-page docs:', summary['multi_page_documents'])
print()
print('Fully text-readable (>=90%):', summary['fully_text_readable'])
print('Image-only (0% text):', summary['image_only_documents'])
print('Needs OCR:', summary['documents_needing_ocr'])
print()

print('=== TEXT-READABLE DOCS ===')
for d in docs:
    cov = d['text'].get('text_coverage', 0)
    if cov > 0 and not d.get('load_error'):
        pages = d['page_count']
        chars = d['text']['total_characters']
        words = d['text']['total_words']
        cls_val = d['classification'].get('document_class',{}).get('value','?')
        langs = '|'.join(d['classification'].get('language_hints',[]))
        curs = '|'.join(d['classification'].get('currency_candidates',[]))
        inv_nums = '|'.join(d['classification'].get('invoice_number_candidates',[]))
        fs = d['classification'].get('financial_structure', {})
        print(f'  {d["filename"]}: pages={pages} cov={cov:.0%} chars={chars} words={words}')
        print(f'    class={cls_val}  langs={langs}  currencies={curs}')
        print(f'    inv_nums={inv_nums}')
        print(f'    fin={fs}')
        print()

print()
print('=== IMAGE-ONLY DOCS ===')
image_only = [d for d in docs if d['text'].get('text_coverage',0) == 0.0 and not d.get('load_error') and d.get('page_count',0) > 0]
for d in image_only:
    all_img = all(p['has_images'] for p in d['pages'])
    print(f'  {d["filename"]}: pages={d["page_count"]} size={d["file_size_bytes"]} bytes  all_have_images={all_img}')

print()
print('=== MULTI-PAGE DOCS (>1 page) ===')
for d in sorted(docs, key=lambda x: x['page_count'], reverse=True):
    if d['page_count'] > 1:
        cov = d['text'].get('text_coverage', 0)
        print(f'  {d["filename"]}: {d["page_count"]} pages  text_coverage={cov:.0%}')

print()
print('=== LARGE FILES (>500KB) ===')
for d in sorted(docs, key=lambda x: x['file_size_bytes'], reverse=True)[:10]:
    print(f'  {d["filename"]}: {d["file_size_bytes"]/1024:.0f} KB, {d["page_count"]} pages')

print()
print('=== DOC PREFIX FAMILIES ===')
from collections import Counter
prefix_counts = Counter(d['filename'].split('-')[0] for d in docs)
for prefix, cnt in prefix_counts.most_common():
    print(f'  {prefix}: {cnt} documents')

print()
print('=== CLASSIFICATION SIGNALS (text-readable only) ===')
for d in docs:
    cov = d['text'].get('text_coverage', 0)
    if cov > 0 and not d.get('load_error'):
        cls = d['classification']
        print(f'  {d["filename"]}:')
        print(f'    doc_class: {cls.get("document_class",{})}')
        print(f'    payable: {cls.get("payable_status",{})}')
        print(f'    master_data_signals: {cls.get("master_data_signals",{})}')
        kw = cls.get('keyword_hits',{})
        print(f'    keyword_hits: {kw}')
        print()
