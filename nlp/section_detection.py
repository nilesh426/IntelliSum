"""Heuristic section detection (never invents sections)."""

import re

_NUMBERED_RE = re.compile(
    r"^\s*(?:(?:\d{1,3}(?:\.\d{1,3}){0,3})\.?|(?:[IVXLC]{1,6})\.?|[A-Z]\.)\s+[A-Z0-9].{2,80}$"
)
_MARKDOWN_RE = re.compile(r"^\s*#{1,4}\s+(.+?)\s*$")


def _is_all_caps_heading(line):
    s = line.strip()
    if len(s) < 4 or len(s) > 90:
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 4:
        return False
    upper = sum(1 for c in letters if c.isupper())
    if upper / len(letters) < 0.85:
        return False
    if s.endswith("."):
        return False
    return True


def _is_title_like(line):
    s = line.strip()
    if len(s) < 3 or len(s) > 80 or s.endswith((".", ",", ";", ":")) is False:
        pass
    if len(s) < 3 or len(s) > 80:
        return False
    if s.endswith((".", "?", "!")):
        return False
    words = s.split()
    if len(words) < 1 or len(words) > 10:
        return False
    # At least half the words start with a capital letter or digit.
    cap = sum(1 for w in words if w[:1].isupper() or w[:1].isdigit())
    return cap / len(words) >= 0.6


def _looks_like_heading(line):
    s = line.strip()
    if not s or len(s) > 100:
        return False
    md = _MARKDOWN_RE.match(line)
    if md and len(md.group(1).strip()) >= 2:
        return True
    if _NUMBERED_RE.match(line):
        return True
    if _is_all_caps_heading(line):
        return True
    # "Something:" short label lines.
    if re.match(r"^[A-Z][\w\s\-/]{2,60}:\s*$", s):
        return True
    return False


def detect_sections(text, headings_hint=None):
    """Detect document sections heuristically.

    Args:
        text: full document text.
        headings_hint: optional list of headings (e.g. from DOCX styles).

    Returns a list of {"title": str, "content": str}.
    Returns [] when sections cannot be reliably detected.
    """
    text = text or ""
    if headings_hint:
        hints = [h.strip() for h in headings_hint if h and h.strip()]
        if len(hints) >= 2:
            sections = []
            # Split text on the hint headings in order of appearance.
            remaining = text
            for i, heading in enumerate(hints):
                idx = remaining.find(heading)
                if idx == -1:
                    continue
                # Content before this heading belongs to previous section.
                before = remaining[:idx]
                if sections and before.strip():
                    sections[-1]["content"] += ("\n\n" + before.strip())
                elif before.strip() and not sections:
                    pass  # preamble before first heading; ignore
                next_idx = None
                if i + 1 < len(hints):
                    nxt = remaining.find(hints[i + 1], idx + len(heading))
                    next_idx = nxt if nxt != -1 else None
                content = (
                    remaining[idx + len(heading): next_idx].strip()
                    if next_idx is not None
                    else remaining[idx + len(heading):].strip()
                )
                sections.append({"title": heading, "content": content})
                if next_idx is None:
                    break
                remaining = remaining[next_idx:]
            # Keep only meaningful sections.
            sections = [s for s in sections if len(s["content"]) >= 30]
            if len(sections) >= 2:
                return sections
            # Fall through to heuristic detection otherwise.

    if not text.strip():
        return []

    lines = text.splitlines()
    heading_positions = []
    for idx, line in enumerate(lines):
        if _looks_like_heading(line):
            title = _MARKDOWN_RE.sub(r"\1", line).strip().strip("#").strip()
            title = re.sub(r"\s+", " ", title)
            heading_positions.append((idx, title))

    # Also consider title-like short lines surrounded by blank lines.
    for idx, line in enumerate(lines):
        if any(pos == idx for pos, _ in heading_positions):
            continue
        if _is_title_like(line):
            prev_blank = idx == 0 or not lines[idx - 1].strip()
            next_blank = idx + 1 >= len(lines) or not lines[idx + 1].strip()
            if prev_blank and (next_blank or idx + 1 < len(lines)):
                # Require the following paragraph to be substantially longer.
                following = " ".join(lines[idx + 1: idx + 4])
                if len(following.strip()) > len(line.strip()) * 2:
                    heading_positions.append((idx, line.strip()))

    heading_positions.sort()
    if len(heading_positions) < 2:
        return []

    sections = []
    for i, (line_idx, title) in enumerate(heading_positions):
        end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(lines)
        content = "\n".join(lines[line_idx + 1: end]).strip()
        if len(content) >= 30:
            sections.append({"title": title, "content": content})

    if len(sections) < 2:
        return []
    return sections
