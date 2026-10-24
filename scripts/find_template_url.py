import re

with open(r'C:\Users\LENOVO\.gemini\antigravity\brain\12ddd2c8-8654-4c8e-98f1-83d2f49f5443\.system_generated\steps\193\content.md', encoding='utf-8') as f:
    text = f.read()

matches = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*template[^<]*)</a>', text, re.I)
for href, label in matches:
    print(f"Label: {label.strip()} -> URL: {href}")

idx = text.find("official template")
if idx != -1:
    print("\nCONTEXT around 'official template':")
    print(text[max(0, idx - 150): min(len(text), idx + 250)])
