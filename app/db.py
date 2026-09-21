"""Database connection, schema creation and seeding."""
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()


def get_db_path() -> str:
    # Read at call time so tests can point to a temporary database.
    return os.getenv("DB_PATH", "curt_inventory.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS parts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    category TEXT NOT NULL,
    location TEXT NOT NULL
);
"""

# (name, quantity, category, location)
SEED_PARTS = [
    # Braking
    ("Brake Pads", 12, "Braking", "Mechanical Workshop"),
    ("Brake Disc", 6, "Braking", "Mechanical Workshop"),
    ("Brake Fluid", 8, "Braking", "Chemical Cabinet"),
    ("Brake Caliper", 2, "Braking", "Mechanical Workshop"),
    # Electronics
    ("ECU", 1, "Electronics", "Electronics Cabinet"),
    ("Wiring Harness", 4, "Electronics", "Electronics Cabinet"),
    ("Wheel Speed Sensor", 10, "Electronics", "Electronics Cabinet"),
    ("Battery Pack", 3, "Electronics", "Battery Storage"),
    # Suspension
    ("Shock Absorber", 8, "Suspension", "Mechanical Workshop"),
    ("Coil Spring", 10, "Suspension", "Mechanical Workshop"),
    ("Control Arm", 4, "Suspension", "Mechanical Workshop"),
    # Drivetrain
    ("Drive Chain", 5, "Drivetrain", "Mechanical Workshop"),
    ("Sprocket", 7, "Drivetrain", "Mechanical Workshop"),
    # Chassis
    ("Wheel Rim", 8, "Chassis", "Tire Rack"),
    ("Tire", 16, "Chassis", "Tire Rack"),
    ("Bolt Kit M8", 40, "Chassis", "Hardware Drawer"),
]


def init_db() -> None:
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def seed_db() -> int:
    """Insert seed parts only if the table is empty. Returns rows inserted."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
    inserted = 0
    if count == 0:
        conn.executemany(
            "INSERT INTO parts (name, quantity, category, location) VALUES (?, ?, ?, ?)",
            SEED_PARTS,
        )
        conn.commit()
        inserted = len(SEED_PARTS)
    conn.close()
    return inserted


if __name__ == "__main__":
    init_db()
    print(f"Seeded {seed_db()} parts into {get_db_path()}")
    