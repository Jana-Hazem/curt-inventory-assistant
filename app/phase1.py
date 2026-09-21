"""Phase 1: rule-based assistant (no AI). Parse the intent first, then call the service."""
import difflib
import re

from app.service import InventoryService

# Words stripped from the START and END of an extracted item name (never the middle,
# so multi-word part names like "Wheel Speed Sensor" stay intact).
FILLER_WORDS = {
    "the", "a", "an", "we", "do", "have", "left", "please", "are", "is", "there",
    "in", "stock", "of", "any", "some", "our", "i", "you", "got", "on", "hand",
    "currently", "available", "stored", "kept", "located", "find", "can", "me",
    "us", "for", "all", "it", "they", "them", "those", "these", "that", "this",
}
# Extra words that are noise when extracting a category name.
CATEGORY_NOISE = {"items", "item", "parts", "part", "stuff", "things", "category",
                  "categories", "everything", "every"}

# Order matters: location -> category -> stock (see parse_intent).
LOCATION_PATTERNS = [re.compile(p) for p in (
    r"\bwhere (?:is|are|can i find|can we find|do i find|do we keep|do we store|would i find)\b(.*)",
    r"\blocation of\b(.*)",
    r"\b(?:is|are)\b(.+?)\b(?:stored|located|kept)\b",
)]
CATEGORY_PATTERNS = [re.compile(p) for p in (
    r"\b(?:list|show|display)\b(.*)",
    r"\bwhat (?:parts |items )?do we have in\b(.*)",
    r"\b(?:all )?(?:items|parts) (?:in|of|under|for)\b(.*)",
    r"\bwhats in\b(.*)",
)]
STOCK_PATTERNS = [re.compile(p) for p in (
    r"\bhow many\b(.*)",
    r"\bhow much\b(.*)",
    r"\b(?:quantity|count|stock|number) of\b(.*)",
    r"\bdo we have\b(.*)",
    r"\bhave we got\b(.*)",
    r"\b(?:is|are) there\b(.*)",
    r"(.*)\bin stock\b",
)]

HELP_MESSAGE = (
    "I can answer three kinds of questions:\n"
    "- Stock: \"How many brake pads do we have?\"\n"
    "- Location: \"Where is the ECU?\"\n"
    "- Category: \"List all items in Electronics\""
)


def _clean(message: str) -> str:
    """Lowercase, drop apostrophes, turn other punctuation into spaces, collapse spaces."""
    text = re.sub(r"['’]", "", message.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_edges(text: str, removable: set[str]) -> str:
    words = text.split()
    while words and words[0] in removable:
        words.pop(0)
    while words and words[-1] in removable:
        words.pop()
    return " ".join(words)


def _resolve_category(text: str, remainder: str, categories: list[str]) -> str | None:
    """Return a canonical category name, the raw unknown text, or None if none was given."""
    # 1. A known category appears as a whole word anywhere in the message.
    for cat in categories:
        c = cat.lower()
        if re.search(rf"\b{re.escape(c)}\b", text):
            return cat
        if c.endswith("s") and re.search(rf"\b{re.escape(c[:-1])}\b", text):
            return cat
    # 2. Otherwise take what follows the trigger and try a close match.
    candidate = _strip_edges(remainder, FILLER_WORDS | CATEGORY_NOISE)
    if not candidate:
        return None
    close = difflib.get_close_matches(candidate, [c.lower() for c in categories], n=1, cutoff=0.8)
    if close:
        return next(c for c in categories if c.lower() == close[0])
    return candidate


def parse_intent(message: str, categories: list[str]) -> dict:
    """Return {"intent": ..., "entity": ...}.

    Intents: empty, location_query, category_query, stock_query, unsupported.
    Order is explicit: 'where ...' is the most specific, and 'what do we have in X'
    must be checked before the stock pattern 'do we have'.
    """
    text = _clean(message)
    if not text:
        return {"intent": "empty", "entity": None}

    for pattern in LOCATION_PATTERNS:
        m = pattern.search(text)
        if m:
            return {"intent": "location_query",
                    "entity": _strip_edges(m.group(1), FILLER_WORDS) or None}

    for pattern in CATEGORY_PATTERNS:
        m = pattern.search(text)
        if m:
            return {"intent": "category_query",
                    "entity": _resolve_category(text, m.group(1), categories)}

    for pattern in STOCK_PATTERNS:
        m = pattern.search(text)
        if m:
            return {"intent": "stock_query",
                    "entity": _strip_edges(m.group(1), FILLER_WORDS) or None}

    return {"intent": "unsupported", "entity": None}


# ---------- response building ----------
def _join_or(items: list[str]) -> str:
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def _is_plural_name(name: str) -> bool:
    return name.lower().endswith("s") and not name.lower().endswith("ss")


def _count_phrase(qty: int, name: str) -> str:
    """'1 ECU' / '2 ECUs' / '1 Brake Pad' / '12 Brake Pads'. Names ending in a digit are kept."""
    if qty == 1 and _is_plural_name(name):
        name = name[:-1]
    elif qty != 1 and name[-1].isalpha() and not _is_plural_name(name):
        name += "es" if name.lower().endswith("ss") else "s"
    return f"{qty} {name}"


def _lookup(entity: str, service: InventoryService):
    """Return (part, note, error_message). Exactly one of part / error_message is set."""
    result = service.find_matches(entity)
    if result["status"] == "found":
        part = result["part"]
        note = ""
        if result["match_type"] in ("partial", "fuzzy"):
            note = f"Showing results for {part['name']}.\n"
        return part, note, None
    if result["status"] == "ambiguous":
        return None, "", f"Did you mean {_join_or(result['suggestions'])}?"
    return None, "", f"I couldn't find '{entity}' in the inventory."


def _answer_category(entity: str | None, service: InventoryService) -> str:
    categories = service.get_categories()
    if entity is None:
        return f"Which category would you like to see? Available: {', '.join(categories)}."
    if entity not in categories:
        return (f"I couldn't find category '{entity}'. "
                f"Available: {', '.join(categories)}.")
    lines = [f"Items in {entity}:"]
    lines += [f"- {p['name']}: {p['quantity']}" for p in service.get_by_category(entity)]
    return "\n".join(lines)


def answer(question: str, service: InventoryService | None = None) -> str:
    service = service or InventoryService()
    parsed = parse_intent(question, service.get_categories())
    intent, entity = parsed["intent"], parsed["entity"]

    if intent == "empty":
        return "Please type a question, for example: \"How many brake pads do we have?\""
    if intent == "unsupported":
        return HELP_MESSAGE
    if intent == "category_query":
        return _answer_category(entity, service)

    if not entity:
        return "Which item would you like to check?"
    part, note, error = _lookup(entity, service)
    if error:
        return error

    name = part["name"]
    if intent == "location_query":
        verb = "are" if _is_plural_name(name) else "is"
        return f"{note}{name} {verb} stored in the {part['location']}."
    # stock_query
    if part["quantity"] == 0:
        return f"{note}We're out of {name} ({part['location']})."
    return f"{note}We have {_count_phrase(part['quantity'], name)} ({part['location']})."


def main() -> None:
    from app.db import init_db, seed_db
    init_db()
    seed_db()
    print("CURT Inventory Assistant (Phase 1). Type 'quit' to exit.")
    while True:
        try:
            question = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"quit", "exit"}:
            break
        print(answer(question), "\n")


if __name__ == "__main__":
    main()