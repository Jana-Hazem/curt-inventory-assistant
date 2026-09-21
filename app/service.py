"""InventoryService: the ONLY place in the project that runs SQL."""
import re
from contextlib import closing
from difflib import SequenceMatcher

from app.db import get_connection

FUZZY_CUTOFF = 0.6        # minimum similarity to be considered a candidate
CLEAR_WINNER_SCORE = 0.8  # top candidate must score at least this...
CLEAR_WINNER_GAP = 0.1    # ...and beat the runner-up by this much to auto-resolve
MAX_SUGGESTIONS = 3


def _singular(word: str) -> str:
    """Very simple plural handling: pads -> pad, but keep 'harness', 'ecu'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse spaces, singularize each word."""
    words = re.sub(r"\s+", " ", text.strip().lower()).split(" ")
    return " ".join(_singular(w) for w in words if w)


def _match(status: str, part: dict | None = None, match_type: str | None = None,
           suggestions: list[str] | None = None) -> dict:
    return {
        "status": status,            # "found" | "ambiguous" | "not_found"
        "part": part,                # the resolved part, when status == "found"
        "match_type": match_type,    # "exact" | "normalized" | "partial" | "fuzzy"
        "suggestions": suggestions or [],
    }


class InventoryService:
    # ---------- basic queries ----------
    def get_part(self, name: str) -> dict | None:
        with closing(get_connection()) as conn:
            row = conn.execute(
                "SELECT * FROM parts WHERE name = ? COLLATE NOCASE", (name.strip(),)
            ).fetchone()
        return dict(row) if row else None

    def get_all_parts(self) -> list[dict]:
        with closing(get_connection()) as conn:
            rows = conn.execute("SELECT * FROM parts ORDER BY category, name").fetchall()
        return [dict(r) for r in rows]

    def get_by_category(self, category: str) -> list[dict]:
        with closing(get_connection()) as conn:
            rows = conn.execute(
                "SELECT * FROM parts WHERE category = ? COLLATE NOCASE ORDER BY name",
                (category.strip(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_categories(self) -> list[str]:
        with closing(get_connection()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT category FROM parts ORDER BY category"
            ).fetchall()
        return [r["category"] for r in rows]

    def part_exists(self, name: str) -> bool:
        return self.get_part(name) is not None

    # ---------- write ----------
    def update_quantity(self, name: str, delta: int) -> dict | None:
        """Add delta (can be negative). Quantity is clamped at 0. None if no such part."""
        if not isinstance(delta, int):
            raise ValueError("delta must be an integer")
        with closing(get_connection()) as conn:
            cur = conn.execute(
                "UPDATE parts SET quantity = MAX(quantity + ?, 0) "
                "WHERE name = ? COLLATE NOCASE",
                (delta, name.strip()),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_part(name)

    # ---------- shared name matching (used by both phases) ----------
    def find_matches(self, name: str) -> dict:
        query = name.strip()
        if not query:
            return _match("not_found")

        parts = self.get_all_parts()

        # 1. Exact, case-insensitive
        for p in parts:
            if p["name"].lower() == query.lower():
                return _match("found", p, "exact")

        # 2. Normalized (case, spaces, simple plurals)
        norm_q = _normalize(query)
        for p in parts:
            if _normalize(p["name"]) == norm_q:
                return _match("found", p, "normalized")

        # 3. Partial: every word typed is a whole word inside the part name
        q_words = set(norm_q.split())
        partial = [p for p in parts if q_words <= set(_normalize(p["name"]).split())]
        if len(partial) == 1:
            return _match("found", partial[0], "partial")
        if len(partial) > 1:
            return _match("ambiguous", suggestions=[p["name"] for p in partial])

        # 4. Fuzzy: difflib similarity (the same ratio get_close_matches uses)
        scored = sorted(
            ((SequenceMatcher(None, norm_q, _normalize(p["name"])).ratio(), p)
             for p in parts),
            key=lambda pair: pair[0],
            reverse=True,
        )
        scored = [pair for pair in scored if pair[0] >= FUZZY_CUTOFF][:MAX_SUGGESTIONS]
        if not scored:
            return _match("not_found")
        if len(scored) == 1:
            return _match("found", scored[0][1], "fuzzy")
        top, runner_up = scored[0][0], scored[1][0]
        if top >= CLEAR_WINNER_SCORE and top - runner_up >= CLEAR_WINNER_GAP:
            return _match("found", scored[0][1], "fuzzy")
        return _match("ambiguous", suggestions=[p["name"] for _, p in scored])