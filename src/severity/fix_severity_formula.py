from pathlib import Path

p = Path(r"src/severity/extract_severity_features.py")
text = p.read_text(encoding="utf-8")

old = 'severity_area = 100.0 * lesion_pixels / leaf_pixels'
if old in text:
    text = text.replace(old, 'leaf_area_total = leaf_pixels + lesion_pixels\n    severity_area = 100.0 * lesion_pixels / leaf_area_total')
else:
    # Match the current script's likely expression.
    old2 = 'lesion_area_percent=100.0 * lesion_pixels / leaf_pixels,'
    if old2 in text:
        text = text.replace(
            old2,
            'lesion_area_percent=100.0 * lesion_pixels / (leaf_pixels + lesion_pixels),'
        )
    else:
        raise RuntimeError("Could not find the severity formula in the script.")

p.write_text(text, encoding="utf-8")
print("Updated src/severity/extract_severity_features.py")
print("New formula: lesion_pixels / (leaf_pixels + lesion_pixels) * 100")
