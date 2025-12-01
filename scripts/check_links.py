#!/usr/bin/env python3
"""
Comprehensive Link Checker for AI Terakoya Knowledge Base
Validates internal links, anchors, and cross-references in HTML files
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Set
from urllib.parse import urlparse, unquote

try:
    from bs4 import BeautifulSoup
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: BeautifulSoup4 or tqdm not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4", "lxml", "tqdm"])
    from bs4 import BeautifulSoup
    import tqdm
    HAS_TQDM = True


class LinkChecker:
    """Comprehensive link validator for HTML files"""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.html_files: List[Path] = []
        self.md_files: List[Path] = []
        self.all_files: Set[Path] = set()
        self.broken_links: List[Dict] = []
        self.missing_anchors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.total_links_checked = 0
        self.file_anchors: Dict[Path, Set[str]] = {}

    def find_files(self):
        """Recursively find all HTML and MD files"""
        print(f"Scanning {self.base_path}...")

        self.html_files = list(self.base_path.rglob("*.html"))
        self.md_files = list(self.base_path.rglob("*.md"))
        self.all_files = set(self.html_files + self.md_files)

        print(f"Found {len(self.html_files)} HTML files")
        print(f"Found {len(self.md_files)} MD files")

    def extract_anchors(self, file_path: Path) -> Set[str]:
        """Extract all anchor IDs from HTML file"""
        if file_path in self.file_anchors:
            return self.file_anchors[file_path]

        anchors = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')

                # Find all elements with id attribute
                for elem in soup.find_all(id=True):
                    anchors.add(elem['id'])

                # Find all <a name="..."> tags (legacy)
                for elem in soup.find_all('a', attrs={'name': True}):
                    anchors.add(elem['name'])

        except Exception as e:
            print(f"Error extracting anchors from {file_path}: {e}")

        self.file_anchors[file_path] = anchors
        return anchors

    def extract_links(self, file_path: Path) -> List[Dict]:
        """Extract all links from HTML or MD file"""
        links = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if file_path.suffix == '.html':
                soup = BeautifulSoup(content, 'html.parser')

                # Remove code blocks to avoid parsing example HTML/links inside them
                for code_block in soup.find_all(['pre', 'code']):
                    code_block.decompose()

                # Extract href links
                for elem in soup.find_all('a', href=True):
                    href = elem['href']
                    # Get line number (approximation)
                    line_num = content[:content.find(str(elem))].count('\n') + 1
                    links.append({
                        'url': href,
                        'line': line_num,
                        'type': 'href',
                        'element': str(elem)[:100]
                    })

                # Extract src links (images, scripts, etc.)
                for tag in ['img', 'script', 'link']:
                    attr = 'src' if tag in ['img', 'script'] else 'href'
                    for elem in soup.find_all(tag, attrs={attr: True}):
                        url = elem[attr]
                        line_num = content[:content.find(str(elem))].count('\n') + 1
                        links.append({
                            'url': url,
                            'line': line_num,
                            'type': attr,
                            'element': str(elem)[:100]
                        })

            elif file_path.suffix == '.md':
                # Remove content inside fenced code blocks before extracting links
                # This prevents false positives from example code (URLs like 'link', 'demo-link', etc.)
                code_block_pattern = r'```[\s\S]*?```'
                content_without_code = re.sub(code_block_pattern, lambda m: '\n' * m.group(0).count('\n'), content)

                # Also remove inline code
                inline_code_pattern = r'`[^`]+`'
                content_without_code = re.sub(inline_code_pattern, '', content_without_code)

                # Extract markdown links [text](url)
                md_link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
                for match in re.finditer(md_link_pattern, content_without_code):
                    url = match.group(2)
                    line_num = content[:match.start()].count('\n') + 1
                    links.append({
                        'url': url,
                        'line': line_num,
                        'type': 'markdown',
                        'element': match.group(0)
                    })

        except Exception as e:
            print(f"Error extracting links from {file_path}: {e}")

        return links

    def is_external_link(self, url: str) -> bool:
        """Check if URL is external"""
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https', 'ftp', 'mailto']

    def resolve_path(self, source_file: Path, url: str) -> Tuple[Path, str]:
        """Resolve relative path to absolute path and extract anchor"""
        # Parse URL and anchor
        if '#' in url:
            path_part, anchor = url.split('#', 1)
        else:
            path_part, anchor = url, ''

        if not path_part:
            # Just an anchor in same file
            return source_file, anchor

        # Decode URL encoding
        path_part = unquote(path_part)

        # Resolve relative path
        source_dir = source_file.parent

        if path_part.startswith('/'):
            # Absolute path from project root
            # Check if it's a GitHub Pages path
            if '/AI-Knowledge-Notes/' in path_part or path_part.startswith('/AI-Knowledge-Notes/'):
                # Extract the path after /AI-Knowledge-Notes/knowledge/en/
                path_part = path_part.replace('/AI-Knowledge-Notes/knowledge/en/', '')
                resolved = self.base_path / path_part
            else:
                # Path from knowledge/en/
                path_part = path_part.lstrip('/')
                resolved = self.base_path / path_part
        else:
            # Relative path
            resolved = (source_dir / path_part).resolve()

        return resolved, anchor

    # SMILES chemical notation patterns to exclude from link checking
    SMILES_PATTERNS = [
        r'^[A-Z]$',                    # Single uppercase letter (C, O, N, etc.)
        r'^=[A-Z]',                    # =O, =N, =S etc.
        r'^\[.*\]$',                   # [OH], [C@@H], [NH2] etc.
        r'^NC\(=O',                    # Amide bond NC(=O
        r'^\[C@@?H?\]',                # Chiral carbon [C@H], [C@@H]
        r'^C\(=O',                     # Carbonyl C(=O
        r'^\w+\d+-\w+',                # Chemical notation like C=C4
        r'^[A-Z][a-z]?\d*$',           # Element with optional number (C, O, N, Na, Ca2)
        r'^[CNOS]=',                   # Double bonds starting with common elements
        r'^\[.*:.*\]$',                # Atom mapping [O:2], [C:1]
        r'^[cnos]\d*$',                # Aromatic atoms (lowercase)
    ]

    def is_smiles_notation(self, url: str) -> bool:
        """Check if the URL looks like SMILES chemical notation"""
        import re
        for pattern in self.SMILES_PATTERNS:
            if re.match(pattern, url):
                return True
        # Additional heuristic: if it contains typical SMILES characters and is short
        smiles_chars = set('=@[]()/#\\')
        if len(url) < 30 and any(c in url for c in smiles_chars):
            # Check if it looks like a chemical formula (has capital letters and special chars)
            if re.match(r'^[A-Za-z0-9=@\[\]()/#\\:,+-]+$', url):
                return True
        return False

    def validate_link(self, source_file: Path, link_info: Dict) -> Dict:
        """Validate a single link"""
        url = link_info['url']

        # Skip SMILES chemical notation (common in chemoinformatics content)
        if self.is_smiles_notation(url):
            return {'status': 'ok', 'reason': 'SMILES notation (skipped)'}

        # Skip external links
        if self.is_external_link(url):
            return {'status': 'external', 'reason': 'External URL (skipped)'}

        # Skip data URIs, javascript, etc.
        if url.startswith(('data:', 'javascript:', 'mailto:', '#')):
            if url == '#' or url.startswith('#'):
                # Anchor-only link in same file
                if url != '#':
                    anchor = url[1:]
                    anchors = self.extract_anchors(source_file)
                    if anchor not in anchors:
                        return {
                            'status': 'missing_anchor',
                            'reason': f'Anchor #{anchor} not found in current file',
                            'target': str(source_file)
                        }
                return {'status': 'ok', 'reason': 'Same-page anchor'}
            return {'status': 'ok', 'reason': 'Special URL (skipped)'}

        try:
            target_file, anchor = self.resolve_path(source_file, url)

            # Check if target file exists
            if not target_file.exists():
                # Try common patterns
                suggestions = []

                # Check if it's missing Dojo prefix (ML/ vs ./ML/)
                if not any(x in url for x in ['/FM/', '/MI/', '/ML/', '/MS/', '/PI/']):
                    for dojo in ['FM', 'MI', 'ML', 'MS', 'PI']:
                        potential = url.replace('./', f'./{dojo}/')
                        potential_path, _ = self.resolve_path(source_file, potential)
                        if potential_path.exists():
                            suggestions.append(potential)

                return {
                    'status': 'broken',
                    'reason': 'File not found',
                    'target': str(target_file),
                    'suggestions': suggestions
                }

            # If there's an anchor, validate it exists
            if anchor:
                anchors = self.extract_anchors(target_file)
                if anchor not in anchors:
                    return {
                        'status': 'missing_anchor',
                        'reason': f'Anchor #{anchor} not found',
                        'target': str(target_file),
                        'anchor': anchor,
                        'available_anchors': sorted(list(anchors))[:10]  # Show first 10
                    }

            return {'status': 'ok', 'reason': 'Valid link'}

        except Exception as e:
            return {
                'status': 'error',
                'reason': f'Error validating link: {str(e)}'
            }

    def check_file(self, file_path: Path):
        """Check all links in a file"""
        links = self.extract_links(file_path)

        for link_info in links:
            self.total_links_checked += 1
            result = self.validate_link(file_path, link_info)

            if result['status'] == 'broken':
                self.broken_links.append({
                    'file': file_path,
                    'line': link_info['line'],
                    'url': link_info['url'],
                    'element': link_info['element'],
                    **result
                })
            elif result['status'] == 'missing_anchor':
                self.missing_anchors.append({
                    'file': file_path,
                    'line': link_info['line'],
                    'url': link_info['url'],
                    'element': link_info['element'],
                    **result
                })
            elif result['status'] == 'error':
                self.warnings.append({
                    'file': file_path,
                    'line': link_info['line'],
                    'url': link_info['url'],
                    **result
                })

    def check_all(self):
        """Check all files"""
        print(f"\nChecking links in {len(self.html_files)} HTML files...")

        files_to_check = self.html_files + self.md_files

        if HAS_TQDM:
            iterator = tqdm.tqdm(files_to_check, desc="Checking files")
        else:
            iterator = files_to_check

        for file_path in iterator:
            self.check_file(file_path)

    def categorize_broken_links(self) -> Dict[str, List]:
        """Categorize broken links by pattern"""
        categories = defaultdict(list)

        for link in self.broken_links:
            url = link['url']

            # Missing Dojo prefix
            if not any(x in url for x in ['/FM/', '/MI/', '/ML/', '/MS/', '/PI/']):
                if url.startswith('./') and '/' in url[2:]:
                    categories['missing_dojo_prefix'].append(link)
                    continue

            # Non-existent series
            if any(series in url for series in ['ml-introduction', 'mi-introduction', 'pi-introduction']):
                categories['non_existent_series'].append(link)
                continue

            # Missing chapter files
            if 'chapter-' in url or 'chapter' in url:
                categories['missing_chapters'].append(link)
                continue

            # Index files
            if 'index.html' in url:
                categories['missing_index'].append(link)
                continue

            # Other
            categories['other'].append(link)

        return dict(categories)

    def generate_report(self, output_file: Path):
        """Generate comprehensive report"""
        categories = self.categorize_broken_links()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Link Checker Report - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            f.write("Summary:\n")
            f.write(f"- Total HTML files: {len(self.html_files)}\n")
            f.write(f"- Total MD files: {len(self.md_files)}\n")
            f.write(f"- Total links checked: {self.total_links_checked}\n")
            f.write(f"- Broken links: {len(self.broken_links)}\n")
            f.write(f"- Missing anchors: {len(self.missing_anchors)}\n")
            f.write(f"- Warnings: {len(self.warnings)}\n\n")

            # Broken Links by Pattern
            if categories:
                f.write("Broken Links by Pattern:\n")
                f.write("-" * 80 + "\n")

                for i, (category, links) in enumerate(categories.items(), 1):
                    category_name = category.replace('_', ' ').title()
                    f.write(f"{i}. {category_name} ({len(links)} instances)\n")

                    # Show example
                    if links:
                        example = links[0]
                        f.write(f"   Example: {example['url']}\n")
                        if example.get('suggestions'):
                            f.write(f"   Suggested fix: {example['suggestions'][0]}\n")
                    f.write("\n")

            # Detailed Report - Broken Links
            if self.broken_links:
                f.write("\n" + "=" * 80 + "\n")
                f.write("Detailed Report: Broken Links\n")
                f.write("-" * 80 + "\n\n")

                for link in sorted(self.broken_links, key=lambda x: str(x['file'])):
                    rel_path = link['file'].relative_to(self.base_path)
                    f.write(f"File: {rel_path}\n")
                    f.write(f"Line {link['line']}: {link['url']}\n")
                    f.write(f"  Status: BROKEN\n")
                    f.write(f"  Reason: {link['reason']}\n")
                    f.write(f"  Target: {link['target']}\n")
                    if link.get('suggestions'):
                        f.write(f"  Suggested fix: {link['suggestions'][0]}\n")
                    f.write("\n")

            # Detailed Report - Missing Anchors
            if self.missing_anchors:
                f.write("\n" + "=" * 80 + "\n")
                f.write("Detailed Report: Missing Anchors\n")
                f.write("-" * 80 + "\n\n")

                for link in sorted(self.missing_anchors, key=lambda x: str(x['file'])):
                    rel_path = link['file'].relative_to(self.base_path)
                    f.write(f"File: {rel_path}\n")
                    f.write(f"Line {link['line']}: {link['url']}\n")
                    f.write(f"  Status: MISSING ANCHOR\n")
                    f.write(f"  Reason: {link['reason']}\n")
                    f.write(f"  Target file exists: {link['target']}\n")
                    if link.get('available_anchors'):
                        f.write(f"  Available anchors: {', '.join(link['available_anchors'][:5])}\n")
                    f.write("\n")

            # Warnings
            if self.warnings:
                f.write("\n" + "=" * 80 + "\n")
                f.write("Warnings and Errors\n")
                f.write("-" * 80 + "\n\n")

                for warning in self.warnings:
                    rel_path = warning['file'].relative_to(self.base_path)
                    f.write(f"File: {rel_path}\n")
                    f.write(f"Line {warning['line']}: {warning['url']}\n")
                    f.write(f"  {warning['reason']}\n\n")

            # Recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("Recommendations:\n")
            f.write("-" * 80 + "\n\n")

            if categories.get('missing_dojo_prefix'):
                f.write(f"1. Fix missing Dojo prefixes ({len(categories['missing_dojo_prefix'])} instances)\n")
                f.write("   - Batch operation recommended\n")
                f.write("   - Pattern: ./series/ → ./ML/series/ (or appropriate Dojo)\n\n")

            if categories.get('non_existent_series'):
                f.write(f"2. Remove or update references to non-existent series ({len(categories['non_existent_series'])} instances)\n")
                f.write("   - These series don't exist in the knowledge base\n\n")

            if categories.get('missing_chapters'):
                f.write(f"3. Complete missing chapter files or remove from navigation ({len(categories['missing_chapters'])} instances)\n\n")

        print(f"\nReport generated: {output_file}")

    def auto_fix(self):
        """Automatically fix common patterns"""
        print("\nAuto-fix mode not yet implemented.")
        print("To fix broken links, use the generated report to guide manual corrections.")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Check links in HTML files')
    parser.add_argument('--path', type=str,
                       default='/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en',
                       help='Base path to scan')
    parser.add_argument('--output', type=str,
                       default='/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/linkcheck_en_local.txt',
                       help='Output report file')
    parser.add_argument('--fix-auto', action='store_true',
                       help='Automatically fix common patterns')

    args = parser.parse_args()

    base_path = Path(args.path)
    output_file = Path(args.output)

    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        sys.exit(1)

    print("AI Terakoya Link Checker")
    print("=" * 80)

    checker = LinkChecker(base_path)
    checker.find_files()
    checker.check_all()
    checker.generate_report(output_file)

    if args.fix_auto:
        checker.auto_fix()

    print("\n" + "=" * 80)
    print(f"Summary: {len(checker.broken_links)} broken links, "
          f"{len(checker.missing_anchors)} missing anchors, "
          f"{len(checker.warnings)} warnings")
    print(f"Total links checked: {checker.total_links_checked}")


if __name__ == '__main__':
    main()
