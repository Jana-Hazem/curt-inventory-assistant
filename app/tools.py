"""Tools the LLM may call. Each tool talks ONLY to InventoryService and returns plain,
JSON-serializable dicts. The model never sees a database connection, only these results."""
import logging

from app.service import InventoryService

logger = logging.getLogger("curt.tools")

LOW_STOCK_THRESHOLD = 3   # quantity at or below this counts as low stock
MAX_ARG_LENGTH = 100      # longest string argument we accept from the model

_service = InventoryService()
_flagged: set[str] = set()   # in-memory, so a restart forgets flags (noted in the README)


# ---------- the three tools ----------
def check_stock(item_name: str) -> dict:
    result = _service.find_matches(item_name)
    if result["status"] != "found":
        return {"found": False, "suggestions": result["suggestions"]}
    part = result["part"]
    return {
        "found": True,
        "name": part["name"],
        "quantity": part["quantity"],
        "location": part["location"],
        "low_stock": part["quantity"] <= LOW_STOCK_THRESHOLD,
        "match_type": result["match_type"],
    }


def list_by_category(category: str) -> dict:
    categories = _service.get_categories()
    canonical = next((c for c in categories if c.lower() == category.strip().lower()), None)
    if canonical is None:
        return {"found": False, "available_categories": categories}
    items = _service.get_by_category(canonical)
    return {
        "found": True,
        "category": canonical,
        "items": [{"name": p["name"], "quantity": p["quantity"], "location": p["location"]}
                  for p in items],
    }


def flag_shortage(item_name: str) -> dict:
    result = _service.find_matches(item_name)
    if result["status"] != "found":
        return {"status": "not_found", "suggestions": result["suggestions"]}
    part = result["part"]
    already = part["name"] in _flagged
    if not already:
        _flagged.add(part["name"])
        logger.warning("Low stock flag for %s (quantity %d)", part["name"], part["quantity"])
    return {"status": "flagged", "item": part["name"], "already_flagged": already}


def reset_flags() -> None:
    """Clear remembered flags (used by tests)."""
    _flagged.clear()


# ---------- registry: the ONLY functions the loop is allowed to execute ----------
TOOL_REGISTRY = {
    "check_stock": check_stock,
    "list_by_category": list_by_category,
    "flag_shortage": flag_shortage,
}
# The one string parameter each tool accepts.
_TOOL_PARAM = {
    "check_stock": "item_name",
    "list_by_category": "category",
    "flag_shortage": "item_name",
}

# Declarations sent to Gemini. The model picks tools from these descriptions, so they matter.
TOOL_DECLARATIONS = [
    {
        "name": "check_stock",
        "description": (
            "Get the current quantity and storage location of a single part by name. "
            "Tolerates small misspellings and partial names. Also says whether stock is low. "
            "If the part is not found or the name is ambiguous, returns suggestions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string",
                              "description": "Name of the part, for example 'Brake Pads'."},
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "list_by_category",
        "description": (
            "List every part in one category with its quantity and location. "
            "If the category does not exist, returns the available categories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string",
                             "description": "Category name, for example 'Electronics'."},
            },
            "required": ["category"],
        },
    },
    {
        "name": "flag_shortage",
        "description": (
            "Record a low-stock shortage flag for a part so the team can reorder it. "
            "Only call this when the user asks to flag a shortage, or agrees to your offer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string",
                              "description": "Name of the part to flag, for example 'ECU'."},
            },
            "required": ["item_name"],
        },
    },
]


# ---------- safe execution ----------
def execute_tool(name: str, args: dict) -> dict:
    """Validate the tool name and arguments, run the tool, and never raise.

    Errors come back as {"error": "..."} so the model can recover and the API never crashes.
    """
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be an object."}

    param = _TOOL_PARAM[name]
    if set(args) != {param}:
        return {"error": f"{name} takes exactly one argument: '{param}'."}
    value = args[param]
    if not isinstance(value, str):
        return {"error": f"'{param}' must be a string."}
    value = value.strip()
    if not value:
        return {"error": f"'{param}' must not be empty."}
    if len(value) > MAX_ARG_LENGTH:
        return {"error": f"'{param}' is too long (max {MAX_ARG_LENGTH} characters)."}

    try:
        return TOOL_REGISTRY[name](value)
    except Exception:  # never leak internals to the model or the user
        logger.exception("Tool %s failed", name)
        return {"error": "The tool failed. Please try again."}