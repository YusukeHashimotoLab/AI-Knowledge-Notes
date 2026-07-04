# PI Exercise Format Conversion

This directory contains scripts to convert PI chapter exercises from plain HTML format to collapsible `<details>` format.

## Files

- **convert_exercises_pi.py** - Main conversion script
- **test_exercise_conversion.py** - Validation test script

## What It Does

The converter performs the following transformations:

1. **Standardizes difficulty labels:**
   - `Basic` → `Easy`
   - `Intermediate` → `Medium`
   - `Advanced` → `Hard`

2. **Converts hint callouts to details:**
   ```html
   <!-- Before -->
   <div class="callout callout-tip">
   <h4>💡 Hint</h4>
   <p>Hint text here...</p>
   </div>

   <!-- After -->
   <details>
   <summary>💡 Hint</summary>
   <p>Hint text here...</p>
   </details>
   ```

3. **Adds hint placeholders for exercises without hints:**
   - Easy: "Think about the basic principles covered in the chapter examples."
   - Medium: "Consider the trade-offs between different approaches and parameter settings."
   - Hard: "Break down the problem into smaller steps and validate each component."

4. **Adds solution placeholders:**
   ```html
   <details>
   <summary>📝 Sample Solution</summary>
   <p><em>Implementation approach:</em></p>
   <ul>
   <li>Step 1: [Key implementation point]</li>
   <li>Step 2: [Analysis or comparison]</li>
   <li>Step 3: [Validation and interpretation]</li>
   </ul>
   </details>
   ```

## Usage

### Test First (Recommended)

Always run the validation test before converting:

```bash
python scripts/test_exercise_conversion.py
```

### Dry Run

See what would be changed without modifying files:

```bash
# All PI files
python scripts/convert_exercises_pi.py --dry-run

# With verbose output
python scripts/convert_exercises_pi.py --dry-run --verbose

# Single file
python scripts/convert_exercises_pi.py --dry-run --file knowledge/en/PI/process-data-analysis/chapter-1.html
```

### Actual Conversion

Run the conversion (creates `.bak` backup files):

```bash
# All PI files
python scripts/convert_exercises_pi.py

# With verbose output
python scripts/convert_exercises_pi.py --verbose

# Single file
python scripts/convert_exercises_pi.py --file knowledge/en/PI/process-data-analysis/chapter-1.html
```

## Safety Features

- **Dry run mode**: Test changes without modifying files
- **Automatic backups**: Creates `.html.bak` files before modification
- **Single file testing**: Test on one file before batch processing
- **Validation tests**: Verify conversion correctness
- **Progress reporting**: Track files processed and changes made

## Example Output

### Before:
```html
<section>
<h2>1.10 Exercises</h2>
<h4>Exercise 1 (Basic): Data Preprocessing</h4>
<p>
  Modify the code from Example 1 and compare the accuracy...
</p>
<h4>Exercise 2 (Intermediate): ARIMA Order Selection</h4>
<p>
  From the ACF/PACF plots in Example 5...
</p>
<div class="callout callout-tip">
<h4>💡 Hint</h4>
<p>For Exercise 3, try different cost functions...</p>
</div>
</section>
```

### After:
```html
<section>
<h2>1.10 Exercises</h2>
<h4>Exercise 1 (Easy): Data Preprocessing</h4>
<p>
  Modify the code from Example 1 and compare the accuracy...
</p>

<details>
<summary>💡 Hint</summary>
<p>Think about the basic principles covered in the chapter examples.</p>
</details>

<details>
<summary>📝 Sample Solution</summary>
<p><em>Implementation approach:</em></p>
<ul>
<li>Step 1: [Key implementation point]</li>
<li>Step 2: [Analysis or comparison]</li>
<li>Step 3: [Validation and interpretation]</li>
</ul>
</details>

<h4>Exercise 2 (Medium): ARIMA Order Selection</h4>
<p>
  From the ACF/PACF plots in Example 5...
</p>

<details>
<summary>💡 Hint</summary>
<p>Consider the trade-offs between different approaches and parameter settings.</p>
</details>

<details>
<summary>📝 Sample Solution</summary>
<p><em>Implementation approach:</em></p>
<ul>
<li>Step 1: [Key implementation point]</li>
<li>Step 2: [Analysis or comparison]</li>
<li>Step 3: [Validation and interpretation]</li>
</ul>
</details>

<details>
<summary>💡 Hint</summary>
<p>For Exercise 3, try different cost functions...</p>
</details>
</section>
```

## Recovery

If you need to revert changes:

```bash
# Restore a single file
mv knowledge/en/PI/process-data-analysis/chapter-1.html.bak knowledge/en/PI/process-data-analysis/chapter-1.html

# Restore all files (from project root)
find knowledge/en/PI -name "*.html.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;
```

## Statistics

Current status (as of last dry run):
- Files scanned: 93
- Files with exercises: 3
- Total exercises: 9
- Existing hints: 3
- Hints to add: 9
- Solutions to add: 9

## Requirements

- Python 3.7+
- No external dependencies (uses standard library only)

## Development

To modify conversion behavior:

1. Edit `convert_exercises_pi.py`
2. Run `test_exercise_conversion.py` to validate
3. Run dry-run to verify on all files
4. Execute actual conversion

## Notes

- The script only processes chapter files (not index files)
- Preserves all HTML structure outside exercise sections
- Maintains proper indentation and formatting
- Safe to run multiple times (idempotent on already-converted files)
