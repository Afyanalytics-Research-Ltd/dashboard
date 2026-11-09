"""v3_passthrough: patients' NOT NULL first_name, and records without a source id."""
import v3_passthrough as pt


def _spec(table):
    key = f"{pt.PREFIX}{table}"
    pt._tables[key] = {"table": table, "facility": "wema_medical", "links": {}, "resolvable": set(), "user_links": []}
    return key


def test_full_name_in_last_name_is_split():
    out = {"last_name": "Jayden Okalo", "first_name": None}
    pt._patient_names(out)
    assert (out["first_name"], out["last_name"]) == ("Jayden", "Okalo")


def test_three_words_and_extra_spaces():
    out = {"last_name": "PAUL  GACOKI  MWANGI", "first_name": ""}
    pt._patient_names(out)
    assert (out["first_name"], out["last_name"]) == ("PAUL", "GACOKI MWANGI")


def test_single_word_becomes_first_name():
    out = {"last_name": "Wanjiku", "first_name": None}
    pt._patient_names(out)
    assert (out["first_name"], out["last_name"]) == ("Wanjiku", "-")


def test_existing_first_name_untouched():
    out = {"last_name": "Mbone", "first_name": "Rebecca"}
    pt._patient_names(out)
    assert (out["first_name"], out["last_name"]) == ("Rebecca", "Mbone")


def test_first_name_without_surname_gets_empty_last_name():
    out = {"last_name": None, "first_name": "Akinyi"}
    pt._patient_names(out)
    assert (out["first_name"], out["last_name"]) == ("Akinyi", "-")


def test_patient_without_any_name_is_held_back():
    key = _spec("patients")
    assert pt.transform({"id": "7", "last_name": None, "first_name": None}, key, {}) is None


def test_reference_defaults_fill_missing_category(monkeypatch):
    made = []
    monkeypatch.setattr(pt, "_reference_id", lambda fac, svc, alias, name: made.append((alias, name)) or 99)
    out = pt.transform({"id": "3", "name": "Paracetamol"}, _spec("inv_products"), {})
    assert out["category_id"] == 99 and out["unit_id"] == 99
    assert ("product_category", "Medicines") in made and ("unit", "Unit") in made
    out = pt.transform({"id": "4", "name": "STOOL O/C", "departmentCode": "Laboratory"}, _spec("procedures"), {})
    assert out["procedure_category_id"] == 99 and ("procedure_categories", "Laboratory") in made


def test_reference_defaults_keep_a_given_value(monkeypatch):
    monkeypatch.setattr(pt, "_reference_id", lambda *a: 99)
    out = pt.transform({"id": "3", "name": "X", "category_id": 7, "unit_id": 8}, _spec("inv_products"), {})
    assert (out["category_id"], out["unit_id"]) == (7, 8)


def test_instance_unique_numbers_come_from_the_source_id():
    out = pt.transform({"id": "12", "status": "paid"}, _spec("invoices"), {"organization_id": 8})
    assert out["invoice_no"] == "WEMA_MEDICAL-org8-INV-12"
    out = pt.transform({"id": "5", "receipt_no": "R-1"}, _spec("payments"), {"organization_id": 8})
    assert out["receipt_no"] == "R-1"                      # a source number is kept


def test_record_without_source_id_is_held_back():
    key = _spec("patients")
    assert pt.transform({"id": None, "last_name": "Jayden Okalo"}, key, {}) is None
    assert pt.transform({"last_name": "Jayden Okalo"}, key, {}) is None


_STORE_UUID = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"


def test_stock_movement_from_store_filled_from_org_cfg():
    key = _spec("inv_stock_movements")
    out = pt.transform({"id": "1", "from_store_id": None}, key, {"default_store": _STORE_UUID})
    assert out["from_store_id"] == _STORE_UUID


def test_stock_movement_from_store_not_overwritten():
    existing = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    key = _spec("inv_stock_movements")
    out = pt.transform({"id": "2", "from_store_id": existing}, key, {"default_store": _STORE_UUID})
    assert out["from_store_id"] == existing


def test_stock_movement_no_default_store_leaves_none():
    key = _spec("inv_stock_movements")
    out = pt.transform({"id": "3", "from_store_id": None}, key, {})
    assert out["from_store_id"] is None
