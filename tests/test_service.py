import pytest

from app import db
from app.service import InventoryService


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    db.seed_db()
    return InventoryService()


def test_get_part_exact(service):
    assert service.get_part("Brake Pads")["quantity"] == 12


def test_get_part_case_insensitive(service):
    assert service.get_part("brake pads")["name"] == "Brake Pads"


def test_get_part_unknown_returns_none(service):
    assert service.get_part("Turbocharger") is None


def test_update_quantity_adds(service):
    assert service.update_quantity("Brake Pads", 3)["quantity"] == 15


def test_update_quantity_never_negative(service):
    assert service.update_quantity("ECU", -100)["quantity"] == 0


def test_update_quantity_unknown_part(service):
    assert service.update_quantity("Turbocharger", 1) is None


def test_category_count(service):
    assert len(service.get_by_category("Braking")) == 4
    assert len(service.get_by_category("braking")) == 4


def test_categories_from_db(service):
    assert set(service.get_categories()) == {
        "Braking", "Electronics", "Suspension", "Drivetrain", "Chassis"
    }


def test_find_exact(service):
    r = service.find_matches("BRAKE PADS")
    assert r["status"] == "found" and r["match_type"] == "exact"


def test_find_plural_normalized(service):
    r = service.find_matches("tires")
    assert r["part"]["name"] == "Tire" and r["match_type"] == "normalized"


def test_find_partial_single(service):
    r = service.find_matches("pads")
    assert r["part"]["name"] == "Brake Pads" and r["match_type"] == "partial"


def test_find_partial_ambiguous(service):
    r = service.find_matches("brake")
    assert r["status"] == "ambiguous"
    assert set(r["suggestions"]) == {"Brake Pads", "Brake Disc", "Brake Fluid", "Brake Caliper"}


def test_find_fuzzy_misspelling(service):
    r = service.find_matches("break pad")
    assert r["status"] == "found"
    assert r["part"]["name"] == "Brake Pads"
    assert r["match_type"] == "fuzzy"


def test_find_unknown(service):
    r = service.find_matches("turbocharger")
    assert r["status"] == "not_found"


def test_find_empty(service):
    assert service.find_matches("   ")["status"] == "not_found"