#!/usr/bin/env python3
"""
HTML Translation Script for Materials Informatics Content

This script translates large HTML files from Japanese to English using Claude API.
It handles files that exceed single-pass token limits by processing them in chunks.

Usage:
    python scripts/translate_html.py <input_file> [--output <output_file>]
"""

import os
import sys
import re
import argparse
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed.")
    print("Install with: pip install anthropic")
    sys.exit(1)


class HTMLTranslator:
    """Translates HTML files from Japanese to English using Claude API."""

    def __init__(self, api_key=None):
        """Initialize translator with API key."""
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it as environment variable or pass as argument."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.glossary = self._load_glossary()

    def _load_glossary(self):
        """Load translation glossary from CSV file."""
        glossary_path = Path(__file__).parent.parent.parent / "claudedocs" / "Translation_Glossary.csv"
        glossary = {}

        if glossary_path.exists():
            with open(glossary_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        japanese, english = parts[0], parts[1]
                        glossary[japanese] = english

        return glossary

    def _extract_html_structure(self, content):
        """Extract HTML structure and identify translatable sections."""
        # Split content into chunks at major section boundaries
        # Look for closing </section>, </div>, or major heading tags
        chunk_pattern = r'(</(?:section|article|div class="[^"]*section[^"]*")>)'
        chunks = re.split(chunk_pattern, content, flags=re.IGNORECASE)

        # Recombine chunks with their closing tags
        result_chunks = []
        for i in range(0, len(chunks)-1, 2):
            if i+1 < len(chunks):
                result_chunks.append(chunks[i] + chunks[i+1])
        if len(chunks) % 2 == 1:
            result_chunks.append(chunks[-1])

        return result_chunks

    def _create_translation_prompt(self, html_chunk, chunk_num, total_chunks):
        """Create translation prompt for a chunk of HTML."""
        glossary_sample = "\n".join([f"- {jp}: {en}" for jp, en in list(self.glossary.items())[:30]])

        return f"""Translate the following HTML content from Japanese to English.

**CRITICAL REQUIREMENTS**:
1. Translate ALL Japanese text to English
2. Preserve ALL HTML tags, attributes, CSS, JavaScript exactly as-is
3. Keep MathJax equations unchanged: \\( ... \\), \\[ ... \\], $$ ... $$
4. Keep code blocks, variable names, function names unchanged
5. Use glossary terms for consistency

**Glossary (key terms)**:
{glossary_sample}
... (see full glossary for complete list)

**Content to translate** (Chunk {chunk_num}/{total_chunks}):
```html
{html_chunk}
```

**Output**: Provide ONLY the translated HTML with no additional explanation or markdown formatting."""

    def translate_chunk(self, html_chunk, chunk_num, total_chunks):
        """Translate a single chunk of HTML using Claude API."""
        prompt = self._create_translation_prompt(html_chunk, chunk_num, total_chunks)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract text content from response
            translated = message.content[0].text

            # Remove markdown code fences if present
            translated = re.sub(r'^```html\s*\n', '', translated)
            translated = re.sub(r'\n```\s*$', '', translated)

            return translated.strip()

        except Exception as e:
            print(f"Error translating chunk {chunk_num}: {e}")
            return html_chunk  # Return original on error

    def translate_file(self, input_path, output_path=None):
        """Translate entire HTML file."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path  # Overwrite original
        else:
            output_path = Path(output_path)

        print(f"Reading file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if file needs translation
        japanese_pattern = r'[一-龯ぁ-んァ-ヶー]'
        japanese_matches = len(re.findall(japanese_pattern, content))

        if japanese_matches == 0:
            print(f"✅ File already fully translated (0 Japanese characters)")
            return

        print(f"📊 Found {japanese_matches} Japanese characters to translate")

        # Extract header and footer
        html_match = re.search(r'(<html[^>]*>)(.*?)(</html>)', content, re.DOTALL)
        if not html_match:
            print("⚠️ Warning: Could not find <html> tags, processing entire file")
            body_content = content
            header = ""
            footer = ""
        else:
            header = html_match.group(1)
            body_content = html_match.group(2)
            footer = html_match.group(3)

        # Update lang attribute to "en"
        header = re.sub(r'lang="ja"', 'lang="en"', header)

        # Split body into chunks
        chunks = self._extract_html_structure(body_content)
        print(f"📦 Split into {len(chunks)} chunks for translation")

        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_japanese = len(re.findall(japanese_pattern, chunk))
            if chunk_japanese == 0:
                print(f"  Chunk {i}/{len(chunks)}: Already in English, skipping")
                translated_chunks.append(chunk)
            else:
                print(f"  Chunk {i}/{len(chunks)}: Translating ({chunk_japanese} Japanese chars)...")
                translated = self.translate_chunk(chunk, i, len(chunks))
                translated_chunks.append(translated)

        # Combine results
        result = header + "".join(translated_chunks) + footer

        # Verify translation
        remaining_japanese = len(re.findall(japanese_pattern, result))
        print(f"\n📊 Translation verification:")
        print(f"   Before: {japanese_matches} Japanese characters")
        print(f"   After:  {remaining_japanese} Japanese characters")
        print(f"   Progress: {100 * (japanese_matches - remaining_japanese) / japanese_matches:.1f}%")

        # Write output
        print(f"\n💾 Writing to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)

        if remaining_japanese == 0:
            print(f"✅ Translation complete! 100% Japanese text removed")
        else:
            print(f"⚠️  Translation incomplete: {remaining_japanese} Japanese characters remaining")
            print(f"   You may need to run the script again or manually review")


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Translate HTML files from Japanese to English using Claude API"
    )
    parser.add_argument(
        'input_file',
        help="Path to input HTML file"
    )
    parser.add_argument(
        '-o', '--output',
        help="Path to output HTML file (defaults to overwriting input)"
    )
    parser.add_argument(
        '--api-key',
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)"
    )

    args = parser.parse_args()

    try:
        translator = HTMLTranslator(api_key=args.api_key)
        translator.translate_file(args.input_file, args.output)
        print("\n🎉 Done!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
