"""Parsing helpers for the standardised [RESPONSE]/[REASONING]/[CONFIDENCE] format."""


def parse_response(content):
    """Split an LLM message that follows the response format into its parts.

    Returns ``(clean_answer, reasoning, confidence)`` where:
      - ``clean_answer`` is the answer text without the format markers,
      - ``reasoning`` is the reasoning text (or None when absent),
      - ``confidence`` is an int when digits are present, otherwise None.
    """
    clean_text = content
    reasoning = None
    confidence = None

    if "[REASONING]" in content:
        parts = content.split("[REASONING]")
        clean_text = parts[0].replace("[RESPONSE]", "").strip()
        rest = parts[1] if len(parts) > 1 else ""
        if "[CONFIDENCE]" in rest:
            meta = rest.split("[CONFIDENCE]")
            reasoning = meta[0].strip() or None
            digits = "".join(filter(str.isdigit, meta[1])).strip()
            confidence = int(digits) if digits else None
        else:
            reasoning = rest.strip() or None

    return clean_text, reasoning, confidence