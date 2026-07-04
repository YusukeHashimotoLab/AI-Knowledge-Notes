#!/usr/bin/env python3
"""
Add Docstrings to Python Code Blocks in HTML Files

Purpose: Scans English HTML files in knowledge/en/ and adds standardized docstrings
         to Python code blocks that have Requirements comments.
Target: Production use
Execution time: ~30-60 seconds for full codebase
Dependencies: beautifulsoup4, lxml
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup, NavigableString
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class CodeBlockInfo:
    """Information extracted from a code block."""
    code: str
    requirements: str
    has_docstring: bool
    context_heading: str
    context_section: str
    file_path: Path
    block_index: int


@dataclass
class DocstringMetadata:
    """Metadata for generating docstrings."""
    title: str
    purpose: str
    target_level: str
    execution_time: str
    dependencies: List[str]


class DocstringGenerator:
    """Generate appropriate docstrings based on code analysis."""

    # Complexity indicators for target level determination
    ADVANCED_KEYWORDS = {
        'multiprocessing', 'asyncio', 'concurrent.futures', 'threading',
        'numba', 'cython', '@jit', 'tensorflow', 'torch', 'keras',
        'sklearn.ensemble', 'optimization', 'gradient', 'neural',
        'quantum', 'distributed', 'cluster', 'GPU', 'CUDA'
    }

    INTERMEDIATE_KEYWORDS = {
        'class ', 'def ', 'lambda', 'decorator', '@', 'yield',
        'generator', 'context manager', 'with ', 'try:', 'except:',
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy'
    }

    # Execution time estimators based on operations
    TIME_INDICATORS = {
        'model.fit': '30-60 seconds',
        'train': '1-5 minutes',
        'optimize': '10-30 seconds',
        'simulation': '5-15 seconds',
        'for ' * 3: '10-20 seconds',  # Nested loops
        'plot': '2-5 seconds',
        'read_csv': '1-3 seconds',
    }

    @staticmethod
    def extract_dependencies(requirements: str) -> List[str]:
        """Extract library names from requirements section."""
        deps = []
        lines = requirements.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # Extract library name (before version specifier)
            match = re.match(r'[-\s]*([a-zA-Z0-9_\-]+)', line)
            if match:
                lib_name = match.group(1)
                if lib_name.lower() not in ['python', 'requirements']:
                    deps.append(lib_name)
        return deps

    @staticmethod
    def determine_target_level(code: str) -> str:
        """Determine target skill level based on code complexity."""
        code_lower = code.lower()

        # Check for advanced patterns
        for keyword in DocstringGenerator.ADVANCED_KEYWORDS:
            if keyword.lower() in code_lower:
                return 'Advanced'

        # Check for intermediate patterns
        intermediate_count = sum(
            1 for keyword in DocstringGenerator.INTERMEDIATE_KEYWORDS
            if keyword.lower() in code_lower
        )

        if intermediate_count >= 3:
            return 'Intermediate'
        elif intermediate_count >= 1:
            return 'Beginner to Intermediate'

        return 'Beginner'

    @staticmethod
    def estimate_execution_time(code: str) -> str:
        """Estimate execution time based on code patterns."""
        code_lower = code.lower()

        # Check for specific time indicators
        for indicator, time_est in DocstringGenerator.TIME_INDICATORS.items():
            if indicator.lower() in code_lower:
                return time_est

        # Count nested loops
        loop_depth = code.count('for ') + code.count('while ')
        if loop_depth >= 3:
            return '10-30 seconds'
        elif loop_depth >= 2:
            return '5-10 seconds'

        # Default based on code length
        lines = len([l for l in code.split('\n') if l.strip()])
        if lines > 100:
            return '10-20 seconds'
        elif lines > 50:
            return '5-10 seconds'

        return '~5 seconds'

    @staticmethod
    def generate_title(context_heading: str, context_section: str, code: str) -> str:
        """Generate appropriate title for the example."""
        # Try to extract meaningful context
        if context_section:
            # Clean up HTML entities and tags
            section = re.sub(r'<[^>]+>', '', context_section)
            section = section.strip()[:60]
            if section:
                return f"Example: {section}"

        if context_heading:
            heading = re.sub(r'<[^>]+>', '', context_heading)
            heading = heading.strip()[:60]
            if heading and heading.lower() not in ['code', 'example', 'implementation']:
                return f"Example: {heading}"

        # Infer from imports
        imports = re.findall(r'import\s+(\w+)', code)
        if 'pandas' in imports:
            return "Example: Data Processing"
        elif 'matplotlib' in imports or 'seaborn' in imports:
            return "Example: Data Visualization"
        elif 'sklearn' in code or 'scikit-learn' in code:
            return "Example: Machine Learning"
        elif 'torch' in imports or 'tensorflow' in imports:
            return "Example: Deep Learning"

        return "Example: Python Implementation"

    @staticmethod
    def generate_purpose(title: str, code: str, dependencies: List[str]) -> str:
        """Generate purpose description based on code analysis."""
        code_lower = code.lower()

        # Pattern-based purpose detection
        if 'plot' in code_lower or 'scatter' in code_lower:
            return "Demonstrate data visualization techniques"
        elif 'fit' in code_lower and ('model' in code_lower or 'sklearn' in code_lower):
            return "Demonstrate machine learning model training and evaluation"
        elif 'dataframe' in code_lower or 'read_csv' in code_lower:
            return "Demonstrate data manipulation and preprocessing"
        elif 'neural' in code_lower or 'layer' in code_lower:
            return "Demonstrate neural network implementation"
        elif 'optimize' in code_lower or 'minimize' in code_lower:
            return "Demonstrate optimization techniques"
        elif 'simulation' in code_lower or 'monte carlo' in code_lower:
            return "Demonstrate simulation and statistical methods"
        elif 'test' in code_lower and 'assert' in code_lower:
            return "Demonstrate testing and validation approaches"

        # Fallback based on dependencies
        if 'numpy' in dependencies:
            return "Demonstrate numerical computation techniques"
        elif 'pandas' in dependencies:
            return "Demonstrate data analysis workflows"

        return "Demonstrate core concepts and implementation patterns"

    @classmethod
    def generate(cls, info: CodeBlockInfo) -> DocstringMetadata:
        """Generate complete docstring metadata for a code block."""
        dependencies = cls.extract_dependencies(info.requirements)
        title = cls.generate_title(info.context_heading, info.context_section, info.code)
        purpose = cls.generate_purpose(title, info.code, dependencies)
        target_level = cls.determine_target_level(info.code)
        execution_time = cls.estimate_execution_time(info.code)

        return DocstringMetadata(
            title=title,
            purpose=purpose,
            target_level=target_level,
            execution_time=execution_time,
            dependencies=dependencies
        )


class HTMLCodeBlockProcessor:
    """Process HTML files and add docstrings to Python code blocks."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            'files_scanned': 0,
            'files_modified': 0,
            'blocks_found': 0,
            'blocks_with_requirements': 0,
            'blocks_already_have_docstring': 0,
            'docstrings_added': 0
        }

    @staticmethod
    def has_docstring(code: str) -> bool:
        """Check if code already has a docstring."""
        # Look for triple quotes near the beginning (after requirements)
        lines = code.split('\n')
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                return True
        return False

    @staticmethod
    def extract_requirements_section(code: str) -> Optional[str]:
        """Extract the Requirements comment section if present."""
        # Look for Requirements section
        req_pattern = r'#\s*Requirements?:\s*\n((?:#.*\n)+)'
        match = re.search(req_pattern, code, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def get_context_info(element) -> Tuple[str, str]:
        """Extract context information from surrounding HTML elements."""
        heading = ""
        section = ""

        # Look for preceding heading tags
        current = element
        for _ in range(10):  # Look up to 10 elements back
            current = current.find_previous(['h1', 'h2', 'h3', 'h4', 'p', 'li'])
            if not current:
                break
            if current.name in ['h1', 'h2', 'h3', 'h4']:
                heading = current.get_text(strip=True)
                break
            elif current.name == 'p' and not section:
                section = current.get_text(strip=True)

        return heading, section

    def process_code_block(self, pre_tag, file_path: Path, block_index: int) -> Optional[CodeBlockInfo]:
        """Process a single code block and extract information."""
        code_tag = pre_tag.find('code', class_='language-python')
        if not code_tag:
            return None

        code = code_tag.get_text()

        # Check for Requirements section
        requirements = self.extract_requirements_section(code)
        if not requirements:
            return None

        self.stats['blocks_with_requirements'] += 1

        # Check if already has docstring
        if self.has_docstring(code):
            self.stats['blocks_already_have_docstring'] += 1
            return None

        # Get context
        heading, section = self.get_context_info(pre_tag)

        return CodeBlockInfo(
            code=code,
            requirements=requirements,
            has_docstring=False,
            context_heading=heading,
            context_section=section,
            file_path=file_path,
            block_index=block_index
        )

    @staticmethod
    def format_docstring(metadata: DocstringMetadata) -> str:
        """Format docstring metadata into proper docstring text."""
        deps_str = ', '.join(metadata.dependencies) if metadata.dependencies else 'None'

        docstring = f'''"""
{metadata.title}

Purpose: {metadata.purpose}
Target: {metadata.target_level}
Execution time: {metadata.execution_time}
Dependencies: {deps_str}
"""'''
        return docstring

    def add_docstring_to_code(self, code: str, docstring: str) -> str:
        """Add docstring to code after the Requirements section."""
        # Find the end of Requirements section
        req_pattern = r'(#\s*Requirements?:\s*\n(?:#.*\n)+)'
        match = re.search(req_pattern, code, re.IGNORECASE)

        if not match:
            return code

        req_end = match.end()

        # Insert docstring with proper spacing
        before = code[:req_end]
        after = code[req_end:]

        # Clean up spacing
        after = after.lstrip('\n')

        return f"{before}\n{docstring}\n\n{after}"

    def process_file(self, file_path: Path) -> int:
        """Process a single HTML file and return number of docstrings added."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'lxml')

            # Find all Python code blocks
            code_blocks = soup.find_all('pre')
            if not code_blocks:
                return 0

            self.stats['blocks_found'] += len(code_blocks)

            modifications = []
            docstrings_added = 0

            for idx, pre_tag in enumerate(code_blocks):
                info = self.process_code_block(pre_tag, file_path, idx)
                if not info:
                    continue

                # Generate docstring
                metadata = DocstringGenerator.generate(info)
                docstring = self.format_docstring(metadata)

                # Add docstring to code
                new_code = self.add_docstring_to_code(info.code, docstring)

                # Store modification
                modifications.append((pre_tag, new_code))
                docstrings_added += 1

                logger.debug(f"  Block {idx}: {metadata.title}")

            if modifications and not self.dry_run:
                # Apply modifications
                for pre_tag, new_code in modifications:
                    code_tag = pre_tag.find('code', class_='language-python')
                    if code_tag:
                        code_tag.string = new_code

                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(soup))

                self.stats['files_modified'] += 1

            self.stats['docstrings_added'] += docstrings_added

            if docstrings_added > 0:
                status = "[DRY RUN]" if self.dry_run else "[MODIFIED]"
                logger.info(f"{status} {file_path.relative_to(file_path.parents[3])}: +{docstrings_added} docstrings")

            return docstrings_added

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return 0

    def process_directory(self, base_path: Path) -> None:
        """Process all HTML files in the knowledge/en/ directory."""
        logger.info(f"Scanning directory: {base_path}")

        if not base_path.exists():
            logger.error(f"Directory not found: {base_path}")
            return

        # Find all HTML files
        html_files = sorted(base_path.glob('**/*.html'))

        if not html_files:
            logger.warning("No HTML files found")
            return

        logger.info(f"Found {len(html_files)} HTML files to process")

        for file_path in html_files:
            self.stats['files_scanned'] += 1
            self.process_file(file_path)

    def print_summary(self) -> None:
        """Print processing summary."""
        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Files scanned:                  {self.stats['files_scanned']}")
        logger.info(f"Files modified:                 {self.stats['files_modified']}")
        logger.info(f"Code blocks found:              {self.stats['blocks_found']}")
        logger.info(f"Blocks with Requirements:       {self.stats['blocks_with_requirements']}")
        logger.info(f"Blocks already have docstring:  {self.stats['blocks_already_have_docstring']}")
        logger.info(f"Docstrings added:               {self.stats['docstrings_added']}")
        logger.info("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Add docstrings to Python code blocks in HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  python add_docstrings.py --dry-run

  # Actually add docstrings
  python add_docstrings.py

  # Process specific directory
  python add_docstrings.py --path knowledge/en/ML

  # Enable debug logging
  python add_docstrings.py --debug
        """
    )

    parser.add_argument(
        '--path',
        type=Path,
        default=Path(__file__).parent.parent / 'knowledge' / 'en',
        help='Base path to scan (default: knowledge/en/)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    # Process files
    processor = HTMLCodeBlockProcessor(dry_run=args.dry_run)
    processor.process_directory(args.path)
    processor.print_summary()

    if args.dry_run:
        logger.info("\nThis was a dry run. Use without --dry-run to apply changes.")

    return 0 if processor.stats['docstrings_added'] >= 0 else 1


if __name__ == '__main__':
    sys.exit(main())
