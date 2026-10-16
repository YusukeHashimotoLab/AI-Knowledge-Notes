/* keyboard-scrollers.js: lets keyboard users reach and scroll boxes that scroll on their own.
 *
 * Canonical copy: knowledge/en/assets/js/keyboard-scrollers.js. knowledge/jp/assets/js/ holds an
 * identical copy; scripts/check_narrow_viewport_css.py checks that the two agree and that every
 * knowledge page loads its locale's copy exactly once (<script defer> before </body>).
 *
 * Wide tables, code, program output and display math scroll inside their own box on narrow screens
 * (knowledge-base.css section 19), and some code blocks do so at every width. Safari (and older
 * Chrome versions) does not stop on such a box when Tab is pressed, so without help a keyboard
 * user cannot scroll it. This script gives every element that is user-scrollable (computed
 * overflow-x or overflow-y auto or scroll) and whose content overflows it tabindex="0": Tab then
 * stops on it and the arrow keys scroll it. It does so whether or not the box contains links or
 * other focusable elements (a focused link does not let every browser scroll its box).
 *
 * Rules
 * - An element gets tabindex="0" only if it has no tabindex attribute, is not focusable by itself
 *   (links, form controls, iframes, ...) and was never handed over to other code (below). The
 *   script marks what it added with data-kbd-scroll.
 * - When a marked element stops overflowing, the script removes its own attribute and mark at the
 *   next evaluation; while the element itself has focus they stay until it loses focus.
 * - If other code writes or removes tabindex on a marked element (even writing the same value),
 *   the element is left to that code for good: the mark is dropped and the script never adds a
 *   tabindex to it again. The script never changes a tabindex it did not add.
 * - mjx-container elements are not touched (MathJax manages their focus, including a reader's
 *   "Include in Tab order" choice in its menu); the inside of mjx-container, .katex, svg/MathML,
 *   script, style, template, noscript and closed <details> is not scanned (a closed <details> is
 *   scanned again when it is opened).
 * - Evaluation runs after the browser has laid out a frame (the first one after two frames),
 *   coalesced, all reads before all writes. It is triggered by DOM changes (child nodes, text,
 *   and the class, style, hidden and open attributes), window resize, <details> toggle, font
 *   loading, image/iframe loads and CSS transitions. If the DOM changes again after the frame,
 *   the evaluation waits for the next frame (at most 3 times, at most 500 ms), so a stop is
 *   never withheld for long.
 * - The script works on its own: it does not wait for MathJax, KaTeX or anything else, and does
 *   nothing in browsers without MutationObserver, WeakSet or requestAnimationFrame.
 */
(function () {
    'use strict';

    if (window.__keyboardScrollers) {
        return;                                    // loaded twice: the first copy does the work
    }
    window.__keyboardScrollers = true;
    if (typeof MutationObserver !== 'function' || typeof WeakSet !== 'function' ||
            typeof window.requestAnimationFrame !== 'function' || !window.getComputedStyle) {
        return;
    }

    var MARK = 'data-kbd-scroll';
    var XHTML = 'http://www.w3.org/1999/xhtml';
    var SKIP = { 'MJX-CONTAINER': 1, 'SCRIPT': 1, 'STYLE': 1, 'TEMPLATE': 1, 'NOSCRIPT': 1 };
    var NATIVE = /^(A|AREA|BUTTON|INPUT|SELECT|TEXTAREA|IFRAME|SUMMARY|AUDIO|VIDEO|EMBED|OBJECT)$/;
    var MAX_DEFERRALS = 3;
    var MAX_WAIT_MS = 500;

    var released = new WeakSet();                  // handed over to other code: never touched again
    var waitingForBlur = new WeakSet();
    var debug = window.__keyboardScrollersDebug;  // set only by tests (measurement hooks)
    var tabObserver = null;
    var pending = false;
    var deferrals = 0;
    var askedAt = 0;
    var generation = 0;
    var generationAtFrame = 0;

    function now() {
        return window.performance && performance.now ? performance.now() : Date.now();
    }

    function scrollable(value) {
        return value === 'auto' || value === 'scroll' || value === 'overlay';
    }

    function overflows(el) {
        var cs = window.getComputedStyle(el);
        return (scrollable(cs.overflowX) && el.scrollWidth > el.clientWidth) ||
               (scrollable(cs.overflowY) && el.scrollHeight > el.clientHeight);
    }

    // Records of tabindex changes that the script did not make: the element is handed over.
    function handleForeign(records) {
        for (var i = 0; i < records.length; i++) {
            var el = records[i].target;
            if (records[i].attributeName === 'tabindex' && el.hasAttribute && el.hasAttribute(MARK)) {
                el.removeAttribute(MARK);
                released.add(el);
                if (debug) { debug.handedOver = (debug.handedOver || 0) + 1; }
            }
        }
    }

    function removeWhenBlurred(el) {
        if (waitingForBlur.has(el)) {
            return;
        }
        waitingForBlur.add(el);
        el.addEventListener('blur', function onBlur() {
            el.removeEventListener('blur', onBlur);
            waitingForBlur['delete'](el);
            schedule();
        });
        if (debug) { debug.keptWhileFocused = (debug.keptWhileFocused || 0) + 1; }
    }

    function evaluate() {
        var t0 = now();
        var body = document.body;
        if (!body) {
            return;
        }
        var add = [];
        var remove = [];
        var check = function (el) {
            if (released.has(el)) {
                return;
            }
            if (el.hasAttribute(MARK)) {
                if (!overflows(el)) {
                    remove.push(el);
                }
            } else if (!el.hasAttribute('tabindex') && !NATIVE.test(el.tagName) &&
                       !el.isContentEditable && overflows(el)) {
                add.push(el);
            }
        };
        var walker = document.createTreeWalker(body, NodeFilter.SHOW_ELEMENT, {
            acceptNode: function (el) {
                if (el.namespaceURI !== XHTML || SKIP[el.tagName] === 1 ||
                        (el.classList && el.classList.contains('katex'))) {
                    return NodeFilter.FILTER_REJECT;
                }
                if (el.tagName === 'DETAILS' && !el.open) {
                    check(el);
                    return NodeFilter.FILTER_REJECT;
                }
                return NodeFilter.FILTER_ACCEPT;
            }
        });
        for (var el = walker.nextNode(); el; el = walker.nextNode()) {
            check(el);
        }
        // Writes. Pending tabindex records were made by other code: handle them first; the records
        // left after the writes below are the script's own and are dropped.
        handleForeign(tabObserver.takeRecords());
        var i, added = 0, removed = 0;
        for (i = 0; i < add.length; i++) {
            if (!released.has(add[i]) && !add[i].hasAttribute('tabindex')) {
                add[i].setAttribute('tabindex', '0');
                add[i].setAttribute(MARK, '');
                added++;
            }
        }
        for (i = 0; i < remove.length; i++) {
            if (!remove[i].hasAttribute(MARK)) {
                continue;
            }
            if (remove[i] === document.activeElement) {
                removeWhenBlurred(remove[i]);
                continue;
            }
            remove[i].removeAttribute('tabindex');
            remove[i].removeAttribute(MARK);
            removed++;
        }
        tabObserver.takeRecords();
        if (debug) {
            var dt = now() - t0;
            debug.evals = (debug.evals || 0) + 1;
            debug.total = (debug.total || 0) + dt;
            debug.max = Math.max(debug.max || 0, dt);
            debug.added = (debug.added || 0) + added;
            debug.removed = (debug.removed || 0) + removed;
            if (debug.first === undefined) { debug.first = dt; debug.firstAt = t0; }
            (debug.log = debug.log || []).push([Math.round(t0), Math.round(dt * 100) / 100, added, removed]);
        }
    }

    function run() {
        if (generation !== generationAtFrame && deferrals < MAX_DEFERRALS && now() - askedAt < MAX_WAIT_MS) {
            deferrals++;                           // the DOM changed after the frame: wait for the next
            if (debug) { debug.deferrals = (debug.deferrals || 0) + 1; }
            frame();
            return;
        }
        pending = false;
        try {
            evaluate();
        } catch (e) {
            if (debug) { debug.error = String(e); }
        }
    }

    var firstFrame = true;

    function frame() {
        // The first evaluation waits one frame more: WebKit can run the first animation frame of a page
        // before its first layout, and reading sizes then would make this script pay for that layout.
        var frames = firstFrame ? 2 : 1;
        firstFrame = false;
        var tick = function () {
            if (--frames > 0) {
                window.requestAnimationFrame(tick);
                return;
            }
            generationAtFrame = generation;
            window.setTimeout(run, 0);             // after this frame's style and layout
        };
        window.requestAnimationFrame(tick);
    }

    function schedule() {
        if (pending) {
            return;
        }
        pending = true;
        deferrals = 0;
        askedAt = now();
        frame();
    }

    function start() {
        var body = document.body;
        tabObserver = new MutationObserver(handleForeign);
        tabObserver.observe(body, { attributes: true, subtree: true, attributeFilter: ['tabindex'],
                                    attributeOldValue: true });
        new MutationObserver(function () {
            generation++;
            schedule();
        }).observe(body, { childList: true, characterData: true, subtree: true, attributes: true,
                           attributeFilter: ['class', 'style', 'hidden', 'open'] });
        window.addEventListener('resize', schedule);
        window.addEventListener('load', schedule);
        document.addEventListener('toggle', schedule, true);        // toggle does not bubble
        document.addEventListener('load', schedule, true);          // images, iframes
        document.addEventListener('transitionend', schedule, true);
        if (document.fonts) {
            if (document.fonts.addEventListener) {
                document.fonts.addEventListener('loadingdone', function () { schedule(); });
            }
            if (document.fonts.ready && document.fonts.ready.then) {
                document.fonts.ready.then(function () { schedule(); });
            }
        }
        schedule();
    }

    if (document.body) {
        start();
    } else {
        document.addEventListener('DOMContentLoaded', start);
    }
}());
