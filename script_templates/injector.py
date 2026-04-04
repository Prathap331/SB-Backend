from __future__ import annotations

from .registry import TEMPLATE_REGISTRY


def assemble_structure_section(template_key: str, target_word_count: int) -> str:
    template = TEMPLATE_REGISTRY[template_key]
    lines = [
        f"SCRIPT STRUCTURE TEMPLATE: {template['name']} ({template['category']})",
        "CRITICAL: Never output segment names, headers, or structural markers in the final script.",
        "These are internal pacing instructions only.",
        "",
    ]
    for seg in template.get("segments", []):
        words = round(float(seg.get("pct", 0.0)) * target_word_count)
        lines.append(f"SEGMENT: {str(seg.get('name','')).upper()} (~{words} words)")
        lines.append("Retrieval directive & generation mode:")
        lines.append(f"  {str(seg.get('brief',''))}")
        lines.append("")
    return "\n".join(lines)


def assemble_chapter_scaffold(chapter_structure: list[dict], duration_minutes: int, wpm: int) -> str:
    lines: list[str] = []
    cumulative_pct = 0.0
    total_words = max(1, duration_minutes * wpm)
    for idx, ch in enumerate(chapter_structure):
        word_pos = int(cumulative_pct * total_words)
        seconds = int((word_pos / max(wpm, 1)) * 60)
        lines.append(
            f"[{seconds//60}:{seconds%60:02d}] Ch {idx+1}: {ch.get('title','')} — {ch.get('covers','')}"
        )
        cumulative_pct += float(ch.get("section_pct", 0.0) or 0.0)
    return "\n".join(lines)
