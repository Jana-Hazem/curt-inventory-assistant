import pytest

from app import db
from app.phase1 import answer, parse_intent
from app.service import InventoryService

CATEGORIES = ["Braking", "Chassis", "Drivetrain", "Electronics", "Suspension"]


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    db.seed_db()
    return InventoryService()


# ---------- intent parsing ----------
STOCK_CASES = [
    ("How many brake pads do we have?", "brake pads"),
    ("how many brake pads are left", "brake pads"),
    ("How much brake fluid is there?", "brake fluid"),
    ("what is the quantity of ECU", "ecu"),
    ("Do we have any sprockets?", "sprockets"),
    ("count of tires", "tires"),
    ("is the battery pack in stock", "battery pack"),
    ("are there any wheel rims left?", "wheel rims"),
    ("how many wheel speed sensors do we have in stock", "wheel speed sensors"),
    ("quantity of brake disc please", "brake disc"),
    ("HOW MANY BRAKE PADS???", "brake pads"),
    ("how many do we have?", None),
]

LOCATION_CASES = [
    ("Where is the ECU?", "ecu"),
    ("where are brake pads stored", "brake pads"),
    ("where can I find the coil springs?", "coil springs"),
    ("location of the drive chain", "drive chain"),
    ("where do we keep tires", "tires"),
    ("where are the pads", "pads"),
    ("Where is turbocharger?", "turbocharger"),
    ("where is the bolt kit m8", "bolt kit m8"),
    ("what is the location of wiring harness", "wiring harness"),
    ("where would I find the control arm", "control arm"),
    ("where are they stored?", None),
]

CATEGORY_CASES = [
    ("list all items in braking", "Braking"),
    ("show electronics", "Electronics"),
    ("What do we have in suspension?", "Suspension"),
    ("list all parts in drivetrain", "Drivetrain"),
    ("show me all chassis parts", "Chassis"),
    ("items in electronics", "Electronics"),
    ("list everything in braking", "Braking"),
    ("what parts do we have in chassis", "Chassis"),
    ("list electronic", "Electronics"),
    ("show me the braking category", "Braking"),
    ("list all items in engine", "engine"),
    ("list all items", None),
]


@pytest.mark.parametrize("message, entity", STOCK_CASES)
def test_stock_intent(message, entity):
    assert parse_intent(message, CATEGORIES) == {"intent": "stock_query", "entity": entity}


@pytest.mark.parametrize("message, entity", LOCATION_CASES)
def test_location_intent(message, entity):
    assert parse_intent(message, CATEGORIES) == {"intent": "location_query", "entity": entity}


@pytest.mark.parametrize("message, entity", CATEGORY_CASES)
def test_category_intent(message, entity):
    assert parse_intent(message, CATEGORIES) == {"intent": "category_query", "entity": entity}


@pytest.mark.parametrize("message", ["what's the weather", "hello", "tell me a joke"])
def test_unsupported(message):
    assert parse_intent(message, CATEGORIES)["intent"] == "unsupported"


@pytest.mark.parametrize("message", ["", "   ", "???"])
def test_empty(message):
    assert parse_intent(message, CATEGORIES)["intent"] == "empty"


# ---------- full answers ----------
def test_stock_answer(service):
    assert answer("How many brake pads do we have?", service) == \
        "We have 12 Brake Pads (Mechanical Workshop)."


def test_singular_and_plural_grammar(service):
    assert answer("how many ecu", service) == "We have 1 ECU (Electronics Cabinet)."
    assert answer("how many tires", service) == "We have 16 Tires (Tire Rack)."
    assert answer("how many bolt kit m8", service) == "We have 40 Bolt Kit M8 (Hardware Drawer)."


def test_out_of_stock(service):
    service.update_quantity("ECU", -5)
    assert answer("how many ecu", service) == "We're out of ECU (Electronics Cabinet)."


def test_location_answer(service):
    assert answer("where is the ecu", service) == "ECU is stored in the Electronics Cabinet."
    assert answer("where are brake pads stored", service) == \
        "Brake Pads are stored in the Mechanical Workshop."


def test_misspelling_shows_corrected_name(service):
    reply = answer("how many break pads", service)
    assert "Showing results for Brake Pads." in reply and "12" in reply


def test_partial_name(service):
    reply = answer("where are the pads", service)
    assert "Showing results for Brake Pads." in reply


def test_ambiguous_item_asks_which(service):
    assert "Did you mean" in answer("how many brake", service)


def test_unknown_item(service):
    assert answer("where is turbocharger", service) == \
        "I couldn't find 'turbocharger' in the inventory."


def test_missing_item_asks_which(service):
    assert answer("how many do we have?", service) == "Which item would you like to check?"


def test_category_answer(service):
    reply = answer("list all items in electronics", service)
    assert "Items in Electronics:" in reply and "- ECU: 1" in reply


def test_unknown_category(service):
    reply = answer("list all items in engine", service)
    assert "couldn't find category 'engine'" in reply and "Braking" in reply


def test_help_and_empty(service):
    assert "three kinds of questions" in answer("what's the weather", service)
    assert "Please type a question" in answer("", service)