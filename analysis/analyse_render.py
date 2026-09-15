"""Analyse the render manifest and check image dimensions."""
import json
from pathlib import Path
from collections import Counter

with open('artifacts/rendered_pages/render_manifest.json', encoding='utf-8') as f:
    m = json.load(f)

s = m['summary']
print('=== RENDER MANIFEST SUMMARY ===')
print('Generated:        ', m['generated_at'])
print('DPI:              ', m['dpi'])
print('Format:           ', m['format'])
print('Total PDFs:       ', s['total_pdfs'])
print('Pages attempted:  ', s['total_pages_attempted'])
print('Pages OK:         ', s['total_pages_ok'])
print('Pages FAILED:     ', s['total_pages_failed'])
print('Total output:     ', s['total_output_mb'], 'MB')
print('Wall time:        ', s['wall_seconds'], 's')
print('Avg/page:         ', s['avg_seconds_per_page'], 's')
print()

print('=== PAGE DIMENSION ANALYSIS ===')
dims = []
for doc in m['documents']:
    for pg in doc['pages']:
        if pg['success']:
            dims.append((pg['source_file'], pg['page_number'], pg['width_px'], pg['height_px']))

widths = [d[2] for d in dims]
heights = [d[3] for d in dims]
print('Width range: ', min(widths), '-', max(widths), 'px')
print('Height range:', min(heights), '-', max(heights), 'px')
print()
print('Most common dimensions (w x h):')
dim_counts = Counter((d[2], d[3]) for d in dims)
for (w, h), cnt in dim_counts.most_common(12):
    print(' ', w, 'x', h, 'px :', cnt, 'pages')

print()
print('=== ORIENTATION ANALYSIS ===')
landscape_pages = [(src, pg, w, h) for src, pg, w, h in dims if w > h]
portrait_pages  = [(src, pg, w, h) for src, pg, w, h in dims if w <= h]
print('Landscape pages:', len(landscape_pages))
print('Portrait pages: ', len(portrait_pages))
for src, pg, w, h in landscape_pages:
    print('  LANDSCAPE:', src, 'page', pg, w, 'x', h)

print()
print('=== SMALL PAGES (<800px in either dim) ===')
small = [(src, pg, w, h) for src, pg, w, h in dims if w < 800 or h < 800]
if small:
    for src, pg, w, h in small:
        print(' ', src, 'page', pg, ':', w, 'x', h)
else:
    print('  None — all pages are >= 800px in both dimensions')

print()
print('=== FILE SIZE DISTRIBUTION ===')
sizes = []
for doc in m['documents']:
    for pg in doc['pages']:
        if pg['success'] and pg['image_path']:
            try:
                sz = Path(pg['image_path']).stat().st_size
                sizes.append((pg['source_file'], pg['page_number'], sz))
            except Exception:
                pass

sizes_kb = [s2[2] / 1024 for s2 in sizes]
print('Per-page size: min=', round(min(sizes_kb)), 'KB, max=', round(max(sizes_kb)), 'KB, avg=', round(sum(sizes_kb)/len(sizes_kb)), 'KB')

print()
print('=== LARGEST PAGES (top 8) ===')
for src, pg, sz in sorted(sizes, key=lambda x: -x[2])[:8]:
    print(' ', src, 'p' + str(pg) + ':', round(sz/1024), 'KB')

print()
print('=== PER-DOCUMENT TIMING (slowest 8) ===')
for doc in sorted(m['documents'], key=lambda d: d['total_render_seconds'], reverse=True)[:8]:
    print(' ', doc['source_file'], str(doc['page_count']) + 'pp,', round(doc['total_render_seconds'], 3), 's,', round(doc['total_bytes']/1024), 'KB')

print()
print('=== TOTAL OUTPUT FILES ===')
all_pngs = list(Path('artifacts/rendered_pages').rglob('*.png'))
print('Total PNG files on disk:', len(all_pngs))
total_disk_bytes = sum(p.stat().st_size for p in all_pngs)
print('Total disk size:', round(total_disk_bytes / 1_048_576, 1), 'MB')
