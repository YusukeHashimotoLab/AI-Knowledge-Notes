#!/usr/bin/env python3
"""
Add Python version and library requirements to code blocks in HTML files.

This script scans English HTML files in knowledge/en/ directories and adds
version requirement comments to Python code blocks with import statements.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import argparse


# Library version specifications (as of 2024)
LIBRARY_VERSIONS: Dict[str, str] = {
    # Core scientific computing
    'numpy': '>=1.24.0, <2.0.0',
    'pandas': '>=2.0.0, <2.2.0',
    'scipy': '>=1.11.0',
    'matplotlib': '>=3.7.0',
    'seaborn': '>=0.12.0',

    # Machine learning
    'scikit-learn': '>=1.3.0, <1.5.0',
    'sklearn': '>=1.3.0, <1.5.0',  # Alias for scikit-learn
    'xgboost': '>=2.0.0',
    'lightgbm': '>=4.0.0',
    'catboost': '>=1.2.0',

    # Deep learning
    'tensorflow': '>=2.13.0, <2.16.0',
    'torch': '>=2.0.0, <2.3.0',
    'keras': '>=3.0.0',
    'pytorch-lightning': '>=2.0.0',

    # NLP
    'transformers': '>=4.30.0',
    'tokenizers': '>=0.13.0',
    'sentencepiece': '>=0.1.99',
    'spacy': '>=3.6.0',
    'nltk': '>=3.8.0',
    'gensim': '>=4.3.0',

    # Computer vision
    'opencv-python': '>=4.8.0',
    'pillow': '>=10.0.0',
    'albumentations': '>=1.3.0',
    'torchvision': '>=0.15.0',

    # Graph/network
    'networkx': '>=3.1.0',
    'graph-tool': '>=2.55',
    'igraph': '>=0.10.0',
    'torch-geometric': '>=2.3.0',
    'dgl': '>=1.1.0',

    # Visualization
    'plotly': '>=5.14.0',
    'bokeh': '>=3.2.0',
    'altair': '>=5.0.0',
    'dash': '>=2.11.0',

    # Data processing
    'polars': '>=0.18.0',
    'pyarrow': '>=12.0.0',
    'dask': '>=2023.5.0',
    'pyspark': '>=3.4.0',

    # Optimization
    'optuna': '>=3.2.0',
    'hyperopt': '>=0.2.7',
    'ray': '>=2.5.0',

    # Time series
    'statsmodels': '>=0.14.0',
    'prophet': '>=1.1.0',
    'pmdarima': '>=2.0.0',

    # ML tools
    'mlflow': '>=2.4.0',
    'wandb': '>=0.15.0',
    'tensorboard': '>=2.13.0',
    'shap': '>=0.42.0',
    'lime': '>=0.2.0',

    # Web/API
    'fastapi': '>=0.100.0',
    'flask': '>=2.3.0',
    'requests': '>=2.31.0',
    'aiohttp': '>=3.8.0',
    'httpx': '>=0.24.0',

    # Utilities
    'tqdm': '>=4.65.0',
    'joblib': '>=1.3.0',
    'pyyaml': '>=6.0.0',
    'python-dotenv': '>=1.0.0',
    'click': '>=8.1.0',

    # Testing
    'pytest': '>=7.4.0',
    'pytest-cov': '>=4.1.0',
    'hypothesis': '>=6.80.0',
}

# Import aliases to actual package names
IMPORT_ALIASES: Dict[str, str] = {
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'yaml': 'pyyaml',
}


class CodeBlockProcessor:
    """Process Python code blocks in HTML files."""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = defaultdict(int)

    def extract_imports(self, code: str) -> Set[str]:
        """
        Extract library names from import statements.

        Args:
            code: Python code string

        Returns:
            Set of library names
        """
        imports = set()

        # Match 'import library' and 'from library import ...'
        import_patterns = [
            r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
        ]

        for line in code.split('\n'):
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    lib = match.group(1)
                    # Map aliases to actual package names
                    lib = IMPORT_ALIASES.get(lib, lib)
                    imports.add(lib)

        return imports

    def has_requirements_comment(self, code: str) -> bool:
        """
        Check if code already has requirements comment.

        Args:
            code: Python code string

        Returns:
            True if requirements comment exists
        """
        # Check for various requirement comment patterns
        patterns = [
            r'#\s*Requirements?:',
            r'#\s*Python\s+\d+\.\d+',
            r'#\s*pip\s+install',
            r'#.*>=\d+\.\d+',
        ]

        first_lines = '\n'.join(code.split('\n')[:10])  # Check first 10 lines

        for pattern in patterns:
            if re.search(pattern, first_lines, re.IGNORECASE):
                return True

        return False

    def generate_requirements_comment(self, imports: Set[str]) -> str:
        """
        Generate requirements comment for imports.

        Args:
            imports: Set of library names

        Returns:
            Requirements comment string
        """
        # Filter to only libraries we have version info for
        versioned_libs = sorted([lib for lib in imports if lib in LIBRARY_VERSIONS])

        if not versioned_libs:
            return ""

        comment_lines = ["# Requirements:", "# - Python 3.9+"]

        for lib in versioned_libs:
            version = LIBRARY_VERSIONS[lib]
            comment_lines.append(f"# - {lib}{version}")

        return '\n'.join(comment_lines) + '\n'

    def process_code_block(self, code: str) -> Tuple[str, bool]:
        """
        Process a single code block, adding requirements if needed.

        Args:
            code: Python code string

        Returns:
            Tuple of (processed_code, was_modified)
        """
        # Skip if already has requirements
        if self.has_requirements_comment(code):
            self.stats['skipped_has_requirements'] += 1
            return code, False

        # Extract imports
        imports = self.extract_imports(code)

        if not imports:
            self.stats['skipped_no_imports'] += 1
            return code, False

        # Generate requirements comment
        requirements = self.generate_requirements_comment(imports)

        if not requirements:
            self.stats['skipped_no_versioned_libs'] += 1
            return code, False

        # Add requirements comment at the beginning
        processed_code = requirements + '\n' + code
        self.stats['modified'] += 1

        return processed_code, True

    def process_html_file(self, file_path: Path) -> bool:
        """
        Process all Python code blocks in an HTML file.

        Args:
            file_path: Path to HTML file

        Returns:
            True if file was modified
        """
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            self.stats['errors'] += 1
            return False

        # Pattern to match Python code blocks
        # Matches: <pre><code class="language-python">...</code></pre>
        pattern = r'(<pre><code\s+class="language-python">)(.*?)(</code></pre>)'

        modified = False
        blocks_in_file = 0

        def replace_code_block(match):
            nonlocal modified, blocks_in_file

            prefix = match.group(1)
            code = match.group(2)
            suffix = match.group(3)

            blocks_in_file += 1

            # HTML decode common entities
            code = code.replace('&lt;', '<')
            code = code.replace('&gt;', '>')
            code = code.replace('&amp;', '&')
            code = code.replace('&quot;', '"')

            # Process code block
            processed_code, was_modified = self.process_code_block(code)

            if was_modified:
                modified = True
                # HTML encode back
                processed_code = processed_code.replace('&', '&amp;')
                processed_code = processed_code.replace('<', '&lt;')
                processed_code = processed_code.replace('>', '&gt;')
                processed_code = processed_code.replace('"', '&quot;')

            return prefix + processed_code + suffix

        # Replace all code blocks
        new_content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)

        if modified and not self.dry_run:
            try:
                file_path.write_text(new_content, encoding='utf-8')
                self.stats['files_modified'] += 1
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                self.stats['errors'] += 1
                return False

        if self.verbose and blocks_in_file > 0:
            status = "MODIFIED" if modified else "unchanged"
            print(f"  {file_path.relative_to(file_path.parents[3])}: {blocks_in_file} blocks ({status})")

        self.stats['files_processed'] += 1
        self.stats['total_code_blocks'] += blocks_in_file

        return modified

    def process_directory(self, base_path: Path) -> None:
        """
        Process all HTML files in knowledge/en/ directories.

        Args:
            base_path: Base path to start searching from
        """
        # Find all HTML files in knowledge/en/ directories
        knowledge_en_path = base_path / 'knowledge' / 'en'

        if not knowledge_en_path.exists():
            print(f"Error: Directory {knowledge_en_path} does not exist")
            sys.exit(1)

        # Get all HTML files
        html_files = sorted(knowledge_en_path.rglob('*.html'))

        print(f"Found {len(html_files)} HTML files to process\n")

        # Process by Dojo category
        dojos = ['FM', 'ML', 'MS', 'MI', 'PI']

        for dojo in dojos:
            dojo_files = [f for f in html_files if f'/{dojo}/' in str(f)]

            if not dojo_files:
                continue

            print(f"\n{'='*60}")
            print(f"Processing {dojo} Dojo ({len(dojo_files)} files)")
            print(f"{'='*60}")

            for file_path in dojo_files:
                self.process_html_file(file_path)

        # Process remaining files not in specific dojos
        other_files = [f for f in html_files
                      if not any(f'/{dojo}/' in str(f) for dojo in dojos)]

        if other_files:
            print(f"\n{'='*60}")
            print(f"Processing Other Files ({len(other_files)} files)")
            print(f"{'='*60}")

            for file_path in other_files:
                self.process_html_file(file_path)

    def print_summary(self) -> None:
        """Print processing summary statistics."""
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed:              {self.stats['files_processed']}")
        print(f"Files modified:               {self.stats['files_modified']}")
        print(f"Total code blocks found:      {self.stats['total_code_blocks']}")
        print(f"Code blocks modified:         {self.stats['modified']}")
        print(f"Skipped (has requirements):   {self.stats['skipped_has_requirements']}")
        print(f"Skipped (no imports):         {self.stats['skipped_no_imports']}")
        print(f"Skipped (no versioned libs):  {self.stats['skipped_no_versioned_libs']}")
        print(f"Errors:                       {self.stats['errors']}")
        print(f"{'='*60}")

        if self.dry_run:
            print("\n*** DRY RUN - No files were actually modified ***")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Add version requirements to Python code blocks in HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  python add_version_info.py --dry-run --verbose

  # Actually modify files
  python add_version_info.py

  # Process specific directory
  python add_version_info.py --base-path /path/to/wp
        """
    )

    parser.add_argument(
        '--base-path',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Base path to wp directory (default: script parent directory)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed processing information'
    )

    args = parser.parse_args()

    # Validate base path
    base_path = args.base_path.resolve()
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist")
        sys.exit(1)

    print(f"Base path: {base_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE MODIFICATION'}")
    print()

    # Process files
    processor = CodeBlockProcessor(dry_run=args.dry_run, verbose=args.verbose)
    processor.process_directory(base_path)
    processor.print_summary()

    # Exit code based on errors
    sys.exit(1 if processor.stats['errors'] > 0 else 0)


if __name__ == '__main__':
    main()
