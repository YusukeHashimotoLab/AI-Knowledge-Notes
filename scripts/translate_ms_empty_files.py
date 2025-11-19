#!/usr/bin/env python3
"""
Translate all empty MS HTML files from Japanese to English
Preserves HTML structure, CSS, JavaScript, MathJax while translating text content
"""

import os
import re
import sys
from pathlib import Path
import anthropic

# Initialize Claude API
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# List of all empty English files to translate
EMPTY_FILES = [
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/3d-printing-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/electrical-magnetic-testing-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/electrical-magnetic-testing-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/electrical-magnetic-testing-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/electrical-magnetic-testing-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/electrical-magnetic-testing-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-science-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-science-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-science-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-science-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-science-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-thermodynamics-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-thermodynamics-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-thermodynamics-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-thermodynamics-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-thermodynamics-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/mechanical-testing-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/mechanical-testing-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/mechanical-testing-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/mechanical-testing-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/polymer-materials-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/polymer-materials-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/polymer-materials-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/polymer-materials-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/processing-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/processing-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/processing-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/processing-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/processing-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/processing-introduction/chapter-6.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-5.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/synthesis-processes-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/synthesis-processes-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/synthesis-processes-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/synthesis-processes-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/thin-film-nano-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/thin-film-nano-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/thin-film-nano-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/thin-film-nano-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/xrd-analysis-introduction/chapter-1.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/xrd-analysis-introduction/chapter-2.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/xrd-analysis-introduction/chapter-3.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/xrd-analysis-introduction/chapter-4.html",
    "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/xrd-analysis-introduction/chapter-5.html",
]

def get_japanese_path(english_path):
    """Convert English path to Japanese path"""
    return english_path.replace('/en/MS/', '/jp/MS/')

def count_japanese_chars(text):
    """Count Japanese characters in text"""
    hiragana = len(re.findall(r'[あ-ん]', text))
    katakana = len(re.findall(r'[ア-ン]', text))
    kanji = len(re.findall(r'[一-龯]', text))
    return hiragana + katakana + kanji

def translate_html(japanese_content, filename):
    """Translate Japanese HTML to English using Claude"""

    prompt = f"""Translate this Japanese HTML file to English. Follow these rules:

1. Translate ALL Japanese text to natural, fluent English
2. Preserve EXACTLY:
   - Complete HTML structure and formatting
   - All CSS styles and classes
   - All JavaScript code
   - All Python code blocks
   - All MathJax equations (keep LaTeX unchanged)
   - All HTML tags and attributes
3. Translate technical terms appropriately for materials science
4. Maintain professional academic tone
5. Keep all URLs, links, and navigation unchanged
6. Output ONLY the complete translated HTML, no explanations

File: {filename}

Japanese HTML:
{japanese_content}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    except Exception as e:
        print(f"ERROR translating {filename}: {e}")
        return None

def main():
    """Main translation workflow"""

    results = {
        'success': [],
        'failed': [],
        'jp_char_counts': {}
    }

    total = len(EMPTY_FILES)

    print(f"Starting translation of {total} files...\n")

    for idx, en_file in enumerate(EMPTY_FILES, 1):
        jp_file = get_japanese_path(en_file)
        filename = Path(en_file).name
        category = Path(en_file).parent.name

        print(f"[{idx}/{total}] {category}/{filename}")

        # Check if Japanese source exists
        if not os.path.exists(jp_file):
            print(f"  ❌ Japanese source not found: {jp_file}")
            results['failed'].append({
                'file': en_file,
                'error': 'Japanese source not found'
            })
            continue

        # Read Japanese content
        try:
            with open(jp_file, 'r', encoding='utf-8') as f:
                jp_content = f.read()
        except Exception as e:
            print(f"  ❌ Error reading Japanese file: {e}")
            results['failed'].append({
                'file': en_file,
                'error': f'Read error: {e}'
            })
            continue

        # Translate
        print(f"  Translating...")
        en_content = translate_html(jp_content, f"{category}/{filename}")

        if en_content is None:
            results['failed'].append({
                'file': en_file,
                'error': 'Translation failed'
            })
            continue

        # Write English content
        try:
            with open(en_file, 'w', encoding='utf-8') as f:
                f.write(en_content)
        except Exception as e:
            print(f"  ❌ Error writing English file: {e}")
            results['failed'].append({
                'file': en_file,
                'error': f'Write error: {e}'
            })
            continue

        # Verify Japanese character count
        jp_count = count_japanese_chars(en_content)
        results['jp_char_counts'][en_file] = jp_count

        if jp_count > 0:
            print(f"  ⚠️  WARNING: {jp_count} Japanese characters remaining")
        else:
            print(f"  ✅ Success (0 Japanese characters)")

        results['success'].append(en_file)

    # Print summary
    print("\n" + "="*80)
    print("TRANSLATION SUMMARY")
    print("="*80)
    print(f"\nTotal files: {total}")
    print(f"Successfully translated: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    if results['failed']:
        print("\nFailed files:")
        for item in results['failed']:
            print(f"  ❌ {item['file']}")
            print(f"     Error: {item['error']}")

    print("\nJapanese character counts:")
    files_with_jp = {k: v for k, v in results['jp_char_counts'].items() if v > 0}
    if files_with_jp:
        print(f"  {len(files_with_jp)} files have remaining Japanese characters:")
        for file, count in sorted(files_with_jp.items(), key=lambda x: x[1], reverse=True):
            print(f"    {count:4d} chars: {file}")
    else:
        print("  ✅ All files have 0 Japanese characters!")

    return 0 if len(results['failed']) == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
