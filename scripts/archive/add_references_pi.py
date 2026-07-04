#!/usr/bin/env python3
"""
Add References section to PI articles missing them.

This script:
1. Scans all PI HTML files
2. Identifies files without References section
3. Adds standard References section before the disclaimer
4. Preserves existing structure and formatting
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Base directory
BASE_DIR = Path(__file__).parent.parent / "knowledge" / "en" / "PI"

# Standard references template for PI articles
REFERENCES_TEMPLATE = """
        <section>
            <h2>References</h2>
            <ol>
                <li>Montgomery, D. C. (2019). <em>Design and Analysis of Experiments</em> (9th ed.). Wiley.</li>
                <li>Box, G. E. P., Hunter, J. S., &amp; Hunter, W. G. (2005). <em>Statistics for Experimenters: Design, Innovation, and Discovery</em> (2nd ed.). Wiley.</li>
                <li>Seborg, D. E., Edgar, T. F., Mellichamp, D. A., &amp; Doyle III, F. J. (2016). <em>Process Dynamics and Control</em> (4th ed.). Wiley.</li>
                <li>McKay, M. D., Beckman, R. J., &amp; Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." <em>Technometrics</em>, 42(1), 55-61.</li>
            </ol>
        </section>
"""


def find_files_without_references(base_dir: Path) -> List[Path]:
    """Find all HTML files in PI directory without References section."""
    files_without_refs = []

    for html_file in base_dir.rglob("*.html"):
        if html_file.is_file():
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if file has References section
            if not re.search(r'<h2[^>]*>References</h2>', content, re.IGNORECASE):
                files_without_refs.append(html_file)

    return files_without_refs


def add_references_section(file_path: Path) -> Tuple[bool, str]:
    """
    Add References section to a file.

    Returns:
        Tuple of (success, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if already has References (shouldn't happen, but double-check)
        if re.search(r'<h2[^>]*>References</h2>', content, re.IGNORECASE):
            return False, "Already has References section"

        # Find insertion point (try multiple patterns in order of preference)
        # Pattern 1: Before disclaimer section (preferred)
        disclaimer_pattern = r'(\s*)(<section class="disclaimer">)'

        # Pattern 2: Before </main> closing tag
        main_close_pattern = r'(\s*)(</main>)'

        # Pattern 3: Before <footer> tag
        footer_pattern = r'(\s*)(<footer>)'

        # Pattern 4: Before </body> tag (last resort)
        body_close_pattern = r'(\s*)(</body>)'

        # Try to insert before disclaimer first (preferred)
        if re.search(disclaimer_pattern, content):
            new_content = re.sub(
                disclaimer_pattern,
                r'\1' + REFERENCES_TEMPLATE.strip() + r'\n\1\2',
                content,
                count=1
            )
        # Otherwise insert before </main>
        elif re.search(main_close_pattern, content):
            new_content = re.sub(
                main_close_pattern,
                r'\1' + REFERENCES_TEMPLATE.strip() + r'\n\1\2',
                content,
                count=1
            )
        # Otherwise insert before <footer>
        elif re.search(footer_pattern, content):
            new_content = re.sub(
                footer_pattern,
                r'\1' + REFERENCES_TEMPLATE.strip() + r'\n\1\2',
                content,
                count=1
            )
        # Last resort: before </body>
        elif re.search(body_close_pattern, content):
            new_content = re.sub(
                body_close_pattern,
                r'\1' + REFERENCES_TEMPLATE.strip() + r'\n\1\2',
                content,
                count=1
            )
        else:
            return False, "Could not find any insertion point"

        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, "References section added successfully"

    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main execution function."""
    print("=" * 70)
    print("PI References Section Addition Script")
    print("=" * 70)

    # Find files without references
    print(f"\nScanning PI directory: {BASE_DIR}")
    files_without_refs = find_files_without_references(BASE_DIR)

    print(f"\nFound {len(files_without_refs)} files without References section")

    if len(files_without_refs) == 0:
        print("All files already have References sections!")
        return

    # Show first 10 files
    print("\nFirst 10 files to update:")
    for i, file_path in enumerate(files_without_refs[:10], 1):
        rel_path = file_path.relative_to(BASE_DIR.parent.parent.parent)
        print(f"  {i}. {rel_path}")

    if len(files_without_refs) > 10:
        print(f"  ... and {len(files_without_refs) - 10} more")

    # Process files (auto-confirm in non-interactive mode)
    print("\nProcessing files...")
    success_count = 0
    fail_count = 0

    for file_path in files_without_refs:
        success, message = add_references_section(file_path)

        if success:
            success_count += 1
            rel_path = file_path.relative_to(BASE_DIR.parent.parent.parent)
            print(f"✅ {rel_path}")
        else:
            fail_count += 1
            rel_path = file_path.relative_to(BASE_DIR.parent.parent.parent)
            print(f"❌ {rel_path}: {message}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  ✅ Successfully updated: {success_count} files")
    print(f"  ❌ Failed: {fail_count} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
