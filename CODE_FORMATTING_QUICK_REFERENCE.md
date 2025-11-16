# Code Formatting Quick Reference

## Correct Format Template

```html
<div class="code-example"><pre><code class="language-python">
# Your code here
import numpy as np

def my_function():
    return "Hello"
</code></pre></div>
```

## Required Components

### 1. Structure
```html
<div class="code-example">        <!-- Container -->
  <pre>                          <!-- Preserves whitespace -->
    <code class="language-X">   <!-- Semantic + highlighting -->
      YOUR CODE HERE
    </code>
  </pre>
</div>
```

### 2. Prism.js in `<head>`
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
```

### 3. Prism.js before `</body>`
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
```

## Common Languages

| Language | Class Name |
|----------|-----------|
| Python | `language-python` |
| JavaScript | `language-javascript` |
| Bash | `language-bash` |
| JSON | `language-json` |

## Scripts

### Fix All Files
```bash
python3 scripts/fix_encoding_and_format.py
```

### Verify All Files
```bash
python3 scripts/verify_code_formatting.py
```

## Common Mistakes

### ❌ Wrong
```html
<div class="code-example">
import numpy as np
</div>
```

### ❌ Wrong
```html
<div class="code-example">
  <pre>
    import numpy as np
  </pre>
</div>
```

### ✅ Correct
```html
<div class="code-example"><pre><code class="language-python">
import numpy as np
</code></pre></div>
```

## Quick Check

1. **Does it have `<pre>`?** ✓
2. **Does it have `<code>`?** ✓
3. **Does `<code>` have language class?** ✓
4. **Is Prism.js included?** ✓

If all ✓, you're good!
