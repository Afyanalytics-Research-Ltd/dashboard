"""Shared test data: a small HMIS-like SQLite database and a V3-style target schema."""

import json
import sqlite3
from pathlib import Path


def make_hmis(path: Path) -> Path:
    c = sqlite3.connect(path)
    c.executescript("""
    create table departments(id integer primary key, name text not null, description text);
    create table patients(id integer primary key, othernames text, surname text, dob date, sex text, phone text, updated_at timestamp);
    create table visits(id integer primary key, patient_id integer references patients(id), department integer, status text, updated_at timestamp);
    insert into departments values (1,'OPD','Outpatient'),(2,'Lab',null);
    insert into patients values (1,'Stephen Ombaka','Murono','1990-01-01','M','0712345678','2026-01-01 10:00:00'),
                                (2,'Jane','Doe','1985-05-05','F',null,'2026-01-02 10:00:00'),
                                (3,'Ali Hassan Omar','Juma','2001-03-03','M','0700000000','2026-01-03 10:00:00');
    insert into visits values (1,1,1,'served','2026-01-01 09:00:00'),(2,2,2,'new','2026-01-02 09:00:00'),(3,3,1,'served','2026-01-03 09:00:00');
    """)
    c.commit()
    c.close()
    return path


TARGET_SCHEMA = {
    "tables": [
        {"name": "patients", "model": "App\\Models\\Reception\\Patient", "columns": [
            {"name": "id", "type": "bigint", "primary_key": True},
            {"name": "first_name", "type": "varchar(100)"}, {"name": "last_name", "type": "varchar(100)"},
            {"name": "date_of_birth", "type": "date"},
            {"name": "gender", "type": "varchar(10)", "enum": ["male", "female"]},
            {"name": "updated_at", "type": "timestamp"}]},
        {"name": "visits", "columns": [
            {"name": "id", "type": "bigint", "primary_key": True},
            {"name": "patient_id", "type": "bigint", "references": "patients.id"},
            {"name": "department_name", "type": "varchar(100)"},
            {"name": "status", "type": "varchar(20)", "enum": ["waiting", "completed"]},
            {"name": "updated_at", "type": "timestamp"}]},
    ]
}


def target_json() -> str:
    return json.dumps(TARGET_SCHEMA)
