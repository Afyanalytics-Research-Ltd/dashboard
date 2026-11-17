import betacare_to_v3 as b


def test_gid_is_stable_per_table_and_not_eagles():
    import eagle_to_v3 as e
    assert b.gid("patients", "9289") == b.gid("patients", " 9289 ") != b.gid("opd", "9289")
    assert b.gid("visit", "1") != e.gid("visit", "1")          # V3 uuids are instance-wide: own namespace


def test_sid():
    assert b.sid("17524") == "17524" and b.sid("0") is None and b.sid("") is None and b.sid(None) is None


def test_method_insurers_and_companies():
    assert b.method_insurers("Cash, Mpesa, NHIF Capitation CS") == ["NHIF Capitation CS"]
    assert b.method_insurers("Cash, Credit-billing error") == [] and b.method_insurers("Unable to pay") == []
    assert b.method_insurers("Mpesa, NHIF, Asterisk Solutions") == ["NHIF", "Asterisk Solutions"]
    assert b.insurer_company("NHIF Edu-Afya") == ("NHIF", "NHIF Edu-Afya")
    assert b.insurer_company("Linda Mama") == ("NHIF", "Linda Mama")
    assert b.insurer_company("Jubilee medical insurance.") == ("Jubilee medical insurance", "Jubilee medical insurance")


def test_key_matches_ward_spellings():
    assert b.key("H.D.U") == "hdu" == b.key("HDU") and b.key("Paediatrics Ward") == "paediatricsward"
    assert b.WARD_MAP[b.key("Female Medical Ward")] == b.key("Female ward")


def test_bp_and_expiry():
    assert b.bp("128/78mmhg") == (128, 78) and b.bp("120 / 75 mmHg") == (120, 75)
    assert b.bp("12/7") == (None, None) and b.bp("") == (None, None) and b.bp("900/80") == (None, 80)
    assert b.month_expiry("Apr/2029") == "2029-04-30" and b.month_expiry("Feb/2028") == "2028-02-29"
    assert b.month_expiry("") is None and b.month_expiry("2027-05-01") == "2027-05-01"


def test_epoch_wall_is_nairobi_time():
    assert b.epoch_wall("1794909798") == "2026-11-17 13:03:18" and b.epoch_wall("0") is None


def test_patient_payload():
    r = {"id": "9288", "patient_unique_id": "946/26", "patient_name": "RUTH WANGUI NYAKIO", "age": "17", "month": "",
         "mobileno": "0768752117", "email": "KAMUCHEGE", "dob": "0000-00-00", "gender": "Female", "address": "",
         "known_allergies": "N/A", "is_active": "yes", "created_at": "2026-11-16 10:44:10"}
    p = b.patient_payload(r, "946/26", None)
    assert (p["first_name"], p["middle_name"], p["last_name"]) == ("RUTH", "WANGUI", "NYAKIO")
    assert p["patient_no"] == "946/26" and p["dob"] is None and p["gender"] == "female" and p["allergies"] is None
    assert p["email"] is None and p["address"] == "KAMUCHEGE" and p["insurance_status"] == "uninsured"
    assert "Age at registration: 17 y" in p["notes"] and p["registration_date"] == "2026-11-16"
    p = b.patient_payload({**r, "email": "a@b.co", "address": "kagaa", "known_allergies": "nkfda"}, "946/26", (1575, "SHA", "SHA"))
    assert p["email"] == "a@b.co" and p["address"] == "kagaa" and p["allergies"] == "nkfda"
    assert p["insurance_status"] == "insured" and p["insurance_company_id"] == 1575


def test_visit_and_vitals_payloads():
    r = {"id": "17524", "patient_id": "9289", "opd_no": "OPDN17524", "appointment_date": "2026-11-17 08:46:00",
         "casualty": "No", "bp": "128/78mmhg", "pulse": "78", "temperature": "366", "weight": "", "note_remark": "review"}
    v = b.visit_payload(r, "opd", 5, facility=94, full_name="Hannah Nduta Gachaka", doctor=7, insured=False)
    assert v["uuid"] == b.gid("opd", "17524") and v["unique_id"] == "BCH-OPDN17524" and v["inpatient"] == 0
    assert v["created_at"] == "2026-11-17 08:46:00" and v["payment_mode"] == "cash" and v["clinic"] == 94
    vt = b.vitals_payload(r, "opd", 1, 5)
    assert vt["blood_pressure"] == "128/78" and vt["pulse"] == 78 and vt["temperature"] == 36.6
    assert "Remarks: review" in vt["notes"]
    assert b.vitals_payload({"id": "1", "appointment_date": "2026-01-01"}, "opd", 1, 5) is None


def test_payment_payload_takes_the_patients_part_only():
    r = {"payment_id": "22176", "method": "Mpesa, SHA", "cash_amount": None, "mpesa_amount": "6500",
         "second_insurance_amount": "10000", "amount": "16500", "timestamp": "1794910577", "date": "2026-11-17"}
    p = b.payment_payload(r, patient=(5, "X Y", "1/26"), visit_id=9, user=512)
    assert p["amount"] == 6500 and p["mpesa_amount"] == 6500 and p["cash_amount"] == 0 and p["payment_mode"] == "mpesa"
    assert p["receipt"] == "BCH-R22176" and p["created_at"].startswith("2026-11-17")
    assert b.payment_payload({**r, "mpesa_amount": None}, patient=None, visit_id=None, user=1) is None
