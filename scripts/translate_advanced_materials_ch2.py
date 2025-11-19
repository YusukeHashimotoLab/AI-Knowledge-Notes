#!/usr/bin/env python3
"""
Comprehensive translation script for Advanced Materials Systems Chapter 2
Translates Japanese HTML to English with structure preservation
"""

import re
import anthropic
import os
from pathlib import Path

def count_japanese_chars(text: str) -> int:
    """Count Japanese characters (Hiragana, Katakana, Kanji)"""
    japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'
    return len(re.findall(japanese_pattern, text))

def translate_with_claude(text: str, client: anthropic.Anthropic) -> str:
    """Translate text using Claude API with context preservation"""

    prompt = f"""Translate this Japanese HTML content to English. Critical requirements:

1. PRESERVE ALL HTML STRUCTURE - tags, attributes, classes, IDs unchanged
2. Translate ALL Japanese text to natural, technical English
3. Keep technical terms accurate (use standard materials science terminology)
4. Maintain mathematical formulas and chemical formulas exactly
5. Keep code blocks unchanged
6. Preserve Mermaid diagram syntax, only translate labels/text inside
7. Keep URLs, file paths unchanged

Japanese text to translate:

{text}

Return ONLY the translated English HTML content, no explanations."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def main():
    # Setup paths
    source_file = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/advanced-materials-systems-introduction/chapter-2.html")
    target_file = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-2.html")

    # Ensure target directory exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Read source
    print(f"Reading source: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count Japanese characters
    total_jp_chars = count_japanese_chars(content)
    total_chars = len(content)
    jp_percentage = (total_jp_chars / total_chars * 100) if total_chars > 0 else 0

    print(f"Total characters: {total_chars:,}")
    print(f"Japanese characters: {total_jp_chars:,} ({jp_percentage:.1f}%)")
    print(f"File lines: {content.count(chr(10)) + 1}")

    # Initialize Claude client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Split content into manageable sections for translation
    # Strategy: Translate in sections based on major HTML structural breaks

    # Section 1: Head and styles (lines 1-380)
    # Section 2: Header and navigation (lines 381-427)
    # Section 3: Content sections (split into logical chunks)

    lines = content.split('\n')

    # Define section boundaries
    sections = []

    # Section 1: HTML head, styles, scripts (mostly unchanged except Japanese text)
    sections.append({
        'name': 'Head and Styles',
        'start': 0,
        'end': 380,
        'translate': True
    })

    # Section 2: Navigation and header
    sections.append({
        'name': 'Navigation and Header',
        'start': 380,
        'end': 427,
        'translate': True
    })

    # Section 3-10: Content sections (split by h2 headers approximately)
    # We'll do content in chunks of ~200-250 lines to stay within token limits
    content_sections = [
        (427, 650, 'Learning objectives and Section 1.1'),
        (650, 900, 'Section 1.1 continuation'),
        (900, 1150, 'Section 1.2'),
        (1150, 1400, 'Section 1.3'),
        (1400, 1650, 'Section 1.4'),
        (1650, 1900, 'Section 1.5'),
        (1900, 2154, 'Final sections and footer'),
    ]

    for start, end, name in content_sections:
        sections.append({
            'name': name,
            'start': start,
            'end': end,
            'translate': True
        })

    # Translate sections
    translated_sections = []

    for i, section in enumerate(sections):
        section_lines = lines[section['start']:section['end']]
        section_text = '\n'.join(section_lines)

        print(f"\n{'='*60}")
        print(f"Section {i+1}/{len(sections)}: {section['name']}")
        print(f"Lines {section['start']}-{section['end']} ({len(section_lines)} lines)")

        jp_chars = count_japanese_chars(section_text)
        print(f"Japanese characters in section: {jp_chars}")

        if section['translate'] and jp_chars > 0:
            print("Translating...")
            translated = translate_with_claude(section_text, client)
            translated_sections.append(translated)

            # Verify translation removed Japanese
            remaining_jp = count_japanese_chars(translated)
            print(f"Remaining Japanese characters: {remaining_jp}")

            if remaining_jp > jp_chars * 0.1:  # If >10% Japanese remains
                print(f"⚠️  Warning: Translation may be incomplete")
        else:
            print("Skipping translation (no Japanese content)")
            translated_sections.append(section_text)

    # Combine all sections
    final_content = '\n'.join(translated_sections)

    # Final verification
    final_jp_chars = count_japanese_chars(final_content)
    final_total_chars = len(final_content)
    final_jp_percentage = (final_jp_chars / final_total_chars * 100) if final_total_chars > 0 else 0

    print(f"\n{'='*60}")
    print("TRANSLATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original Japanese chars: {total_jp_chars:,} ({jp_percentage:.1f}%)")
    print(f"Final Japanese chars: {final_jp_chars:,} ({final_jp_percentage:.1f}%)")
    print(f"Translation success: {((total_jp_chars - final_jp_chars) / total_jp_chars * 100):.1f}%")

    # Write output
    print(f"\nWriting to: {target_file}")
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print("✅ Translation complete!")
    print(f"\nStatistics:")
    print(f"  Source file: {source_file}")
    print(f"  Target file: {target_file}")
    print(f"  Original Japanese: {total_jp_chars:,} chars ({jp_percentage:.1f}%)")
    print(f"  Remaining Japanese: {final_jp_chars:,} chars ({final_jp_percentage:.1f}%)")

if __name__ == "__main__":
    main()
