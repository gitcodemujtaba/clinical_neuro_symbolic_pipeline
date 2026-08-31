"""ui/components/keyboard_shortcuts.py -- 2026-08-31: optional keyboard
shortcuts for the HITL review page, the last of the eight identified
review-throughput/context gaps.

WHY THIS IS RAW JS, NOT A THIRD-PARTY PACKAGE. No keyboard-shortcut
package (e.g. streamlit-shortcuts) is installed in this environment, and
adding an undeclared new dependency to requirements.txt for one small
feature isn't warranted -- this project has no precedent of pulling in a
UI-framework package for something a ~30-line script can do directly.
Streamlit buttons/radios are real DOM elements; a plain keydown listener
that finds and .click()s them works without any Python<->JS state
synchronization (Streamlit's own frontend handles the resulting click
exactly like a real user click).

WHY THIS IS OPT-IN, DEFAULT OFF. Unlike every other addition this
session, DOM-selector-based JS is fundamentally more fragile than pure
Python -- it depends on Streamlit's current data-testid attributes and
visible button/radio text, which could silently break on a Streamlit
version upgrade, and it CANNOT be covered by this project's own
Python test suite the way every other change this session was verified
(no automated check exists here beyond "does it inject without a Python
exception" -- the actual keyboard behavior needs manual/visual
confirmation, stated honestly as a real testing gap, not glossed over).
Gating it behind an explicit checkbox means a reviewer who doesn't want
it (or hits a version where it misbehaves) is never affected by it.

KEYS: 1/2/3 = Approve/Correct/Reject (sets the decision radio, does NOT
auto-submit -- a reviewer still confirms via the Submit button, so a
mis-fired keystroke can't silently submit a wrong decision). Left/Right
arrows = Previous/Next case. All shortcuts are suppressed while focus is
inside any text input/textarea/contenteditable element, so typing in the
concept-search box or the comment field is never hijacked -- the single
most important correctness property of this script.
"""
import streamlit as st

_SCRIPT = """
<script>
(function() {
    // st.components.v1.html() renders THIS script inside its own iframe --
    // this iframe is a zero-height, never-focused element, so listening
    // on ITS OWN `document` would never see a real keystroke (the user's
    // focus is always on the actual page, outside the iframe). Both the
    // listener AND the duplicate-attach guard must live on the PARENT
    // page's window/document, which persists across Streamlit reruns
    // (this iframe itself is typically re-created on every rerun, so a
    // guard flag on the iframe's own `window` would never actually guard
    // anything).
    const parentWin = window.parent;
    if (parentWin.__hitl_shortcuts_attached) { return; }
    parentWin.__hitl_shortcuts_attached = true;
    const doc = parentWin.document;

    function isTypingTarget(el) {
        if (!el) return false;
        const tag = (el.tagName || "").toLowerCase();
        return tag === "input" || tag === "textarea" || el.isContentEditable;
    }

    function clickButtonByText(text) {
        const buttons = doc.querySelectorAll('[data-testid="stButton"] button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"]');
        for (const b of buttons) {
            if ((b.textContent || "").trim() === text) { b.click(); return true; }
        }
        return false;
    }

    function clickRadioByLabel(groupHint, optionText) {
        // Scope to the FIRST stRadio widget whose own option set includes
        // groupHint (e.g. "APPROVED") -- this page has only one such
        // radio (the decision selector), but scoping defensively avoids
        // ever touching an unrelated radio if the page's layout changes.
        const groups = doc.querySelectorAll('[data-testid="stRadio"]');
        for (const g of groups) {
            const labels = g.querySelectorAll('label');
            let hasHint = false;
            for (const l of labels) { if ((l.textContent||"").includes(groupHint)) { hasHint = true; break; } }
            if (!hasHint) continue;
            for (const l of labels) {
                if ((l.textContent || "").trim() === optionText) {
                    const input = l.querySelector('input');
                    if (input) { input.click(); return true; }
                }
            }
        }
        return false;
    }

    doc.addEventListener('keydown', function(ev) {
        if (isTypingTarget(ev.target)) { return; }
        if (ev.key === '1') { clickRadioByLabel('APPROVED', 'APPROVED'); }
        else if (ev.key === '2') { clickRadioByLabel('APPROVED', 'CORRECTED'); }
        else if (ev.key === '3') { clickRadioByLabel('APPROVED', 'REJECTED'); }
        else if (ev.key === 'ArrowLeft') { clickButtonByText('← Previous'); }
        else if (ev.key === 'ArrowRight') { clickButtonByText('Next →'); }
    });
})();
</script>
"""


def render_keyboard_shortcuts_toggle():
    """Renders the opt-in checkbox + (if enabled) injects the listener
    script. Call once per page render; safe to call unconditionally --
    the script itself no-ops on a second injection (a guard flag on the
    PARENT page's window, not this component's own iframe window, since
    the iframe is recreated on every rerun but the parent page persists).

    KNOWN LIMITATION, disclosed rather than silently broken: once enabled
    in a browser tab, unchecking this box stops INJECTING the script
    again but does not detach the already-attached listener (there is no
    teardown call here) -- shortcuts stay live until the page is
    reloaded. Acceptable for an experimental, opt-in feature; stated in
    the checkbox's own help text so it's not a silent surprise.
    """
    enabled = st.sidebar.checkbox(
        "⌨️ Keyboard shortcuts (experimental)", value=False,
        help="1/2/3 = Approve/Correct/Reject (sets the decision, does not "
             "auto-submit). ←/→ = Previous/Next case. Suppressed while "
             "typing in any text field. DOM-based, not covered by this "
             "project's automated tests. Known limitation: once enabled, "
             "un-checking this box does NOT detach the listener in this "
             "browser tab -- reload the page to fully turn it off.")
    if enabled:
        st.components.v1.html(_SCRIPT, height=0)
    return enabled
