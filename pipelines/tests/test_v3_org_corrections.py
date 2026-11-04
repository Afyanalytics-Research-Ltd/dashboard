"""v3_org_corrections pure helpers + the pipeline fixes of 2026-10-17."""
import v3_org_corrections as oc


def test_latest_per_id_keeps_one_row_per_v2_id_latest_edit():
    rows = [{"id": 1, "updated_at": "2026-01-01 00:00:00", "x": "old"},
            {"id": 1, "updated_at": "2026-02-01 00:00:00", "x": "new"},
            {"id": 2, "updated_at": None, "x": "only"}]
    out = {r["id"]: r["x"] for r in oc.latest_per_id(rows)}
    assert out == {1: "new", 2: "only"}


def test_vd_uuid_is_stable_and_per_facility():
    assert oc.vd_uuid("kisumu_v3", 46) == oc.vd_uuid("kisumu_v3", "46")
    assert oc.vd_uuid("kisumu_v3", 46) != oc.vd_uuid("other", 46)


def test_vd_payload_status_and_times():
    r = {"department": "Triage", "checkout": 1, "finish_at": None, "begin_at": "",
         "created_at": "2026-08-14T14:09:29", "updated_at": "2026-08-14 14:09:29", "deleted_at": None}
    p = oc.vd_payload(r, 10, 6149, None, "u")
    assert p["status"] == "completed" and p["checkout"] == 1 and p["destination_id"] == 6149
    assert p["created_at"] == "2026-08-14 14:09:29" and p["begin_at"] is None
    q = oc.vd_payload({"checkout": 0}, 10, None, None, "u", "Lab")
    assert q["status"] == "pending" and q["department"] == "Lab"


def test_v2_admission_by_number_then_visit():
    v2 = {"7340": {"id": 7340, "admitted_at": "2024-09-02 18:00:00"}}
    by_visit = {5: [{"id": 1, "admitted_at": "2024-01-01 00:00:00"}, {"id": 2, "admitted_at": "2024-03-01 00:00:00"}]}
    assert oc.v2_admission_for({"admission_number": "ADM-7340.284"}, v2, by_visit)["id"] == 7340
    near = oc.v2_admission_for({"admission_number": "X", "visit_id": 5, "admission_date": "2024-02-25"}, v2, by_visit)
    assert near["id"] == 2
    assert oc.v2_admission_for({"admission_number": "ADM-1", "visit_id": 9}, v2, by_visit) is None


def test_discharge_plan_discharged_closes_open_stays():
    ad = {"id": 1, "status": "admitted", "discharge_date": None}
    stays = [{"id": 10, "allocation_date": "2024-01-01", "discharge_date": None, "transfer_out_date": None,
              "release_date": None}]
    adm, stay, need = oc.discharge_plan(ad, {"discharged_at": "2024-01-05 10:00:00"}, stays)
    assert adm == [(1, {"status": "discharged", "discharge_date": "2024-01-05 10:00:00"})]
    assert stay[0][0] == 10 and stay[0][1]["discharge_date"] == "2024-01-05 10:00:00" and not need


def test_discharge_plan_still_in_reopens_latest_stay_or_needs_one():
    ad = {"id": 1, "status": "discharged", "discharge_date": "2024-01-05"}
    stays = [{"id": 10, "allocation_date": "2024-01-01", "discharge_date": "2024-01-02"},
             {"id": 11, "allocation_date": "2024-01-03", "discharge_date": "2024-01-05"}]
    adm, stay, need = oc.discharge_plan(ad, {"discharged_at": None}, stays)
    assert adm == [(1, {"status": "admitted", "discharge_date": None})]
    assert [s[0] for s in stay] == [11] and not need
    assert oc.discharge_plan({"id": 2, "status": "admitted"}, {"discharged_at": ""}, [])[2] is True


def test_discharge_plan_nothing_when_already_right():
    ad = {"id": 1, "status": "discharged", "discharge_date": "2024-01-05 10:00:00"}
    stays = [{"id": 10, "discharge_date": "2024-01-05 10:00:00"}]
    assert oc.discharge_plan(ad, {"discharged_at": "2024-01-05 10:00:00"}, stays) == ([], [], False)


def test_category_ids_by_name():
    v2cat = {"1": "Doctor", "6": "Nursing", "99": "Gone"}
    new, missing = oc.category_ids("1,6,99,1", v2cat, {"doctor": "30643", "nursing": "30669"})
    assert new == "30643,30669" and missing == ["Gone"]
    assert oc.category_ids("", v2cat, {}) == (None, [])


def test_facility_sql_has_before_update_after():
    sql = oc.facility_sql(284, 283, [{"service": "reception", "table": "patients", "wrong": {247: 53237}}])
    assert "UPDATE patients SET facility_id = 283 WHERE organization_id = 284 AND facility_id IN (247);" in sql
    assert "'before'" in sql and "'after'" in sql and "53237 AS expected" in sql and "START TRANSACTION;" in sql


def test_admission_transform_sets_discharge_state():
    import snowflake_to_v3_migration  # noqa: F401  (registers the mappings)
    import v2_to_v3_api_migration as m
    inj = m._PER_KEY_INJECT["inpatient_admission"]
    assert inj["status"]({"discharged_at": "2024-10-12 14:39:27"}) == "discharged"
    assert inj["discharge_date"]({"discharged_at": "2024-10-12 14:39:27"}) == "2024-10-12 14:39:27"
    assert inj["status"]({"discharged_at": None}) == "admitted" and inj["discharge_date"]({}) is None


def test_destination_categories_fk_list_mapping():
    import snowflake_to_v3_migration  # noqa: F401
    import v2_to_v3_api_migration as m
    f = m._PER_KEY_COERCIONS["gen_settings_destinations"]["associated_procedure_category"]
    m._id_map[m._v3_alias(r"App\Models\ProcedureCategory")] = {1: 501, 6: 506}
    assert f("1,6,46") == "501,506" and f("") is None


def test_eval_visit_destinations_link_rules():
    import v3_to_v3_links as L
    assert L.MODEL_LINKS["evaluation/visit_destinations"] == {"clinic_id": L.DROP, "room_id": L.DROP}
    assert L.LINKS["destination_id"] == "core/service_destinations"
    assert L.critical_cols("evaluation/visit_destinations") == ("visit_id",)


def test_cons_payload_keeps_v2_dates_and_flags():
    r = {"user_id": "1490", "quantity": "2", "discount": "0.00", "price": "42.00", "amount": "84.00", "bill": 1,
         "cancelled": 0, "created_at": "2026-10-12 10:31:05", "updated_at": "2026-10-12T10:31:05",
         "date_adjusted_from": "2026-10-01 00:00:00", "consumable_type_name": "GLOVES"}
    p = oc.cons_payload(r, 10, 5751975, lambda v: {1490: 7}.get(int(v)) if v else None, 215, None, "u")
    assert p["visit_id"] == 10 and p["product_id"] == 5751975 and p["user_id"] == 7 and p["invoice_id"] == 215
    assert p["created_at"] == "2026-10-12 10:31:05" and p["updated_at"] == "2026-10-12 10:31:05"
    assert p["bill"] == 1 and p["cancelled"] == 0 and p["status"] == "billed" and p["date_adjusted_from"] == "2026-10-01"
    c = oc.cons_payload({"cancelled": "1"}, 10, None, lambda v: None, None, None, "u")
    assert c["status"] == "cancelled" and c["cancelled"] == 1 and c["quantity"] is None


def test_cons_payload_store_product():
    p = oc.cons_payload({}, 10, 5748332, lambda v: None, None, None, "u", 5748332)
    assert p["store_product_id"] == 5748332 and p["product_id"] == 5748332
    assert oc.cons_payload({}, 10, 1, lambda v: None, None, None, "u")["store_product_id"] is None


def test_v2_user_lookup_uuid5_then_placeholder_then_unique_email_or_username():
    import uuid
    u5 = str(uuid.uuid5(oc.USER_UUID_NS, "kisumu_v3/users/1430"))
    v3 = [{"id": 3838, "uuid": u5, "email": "odira71@yahoo.com", "username": "fodira"},
          {"id": 2236, "uuid": "x", "email": "odira71@yahoo.com.v2-1430.invalid", "username": "fodira2"},
          {"id": 2277, "uuid": "y", "email": "lina@yahoo.com.v2-1457.invalid", "username": "lakelo"},
          {"id": 4180, "uuid": "z", "email": "Achola@Gmail.com", "username": "achola"},
          {"id": 5001, "uuid": "a", "email": "same@x.com", "username": "dup"},
          {"id": 5002, "uuid": "b", "email": "same@x.com", "username": "dup"}]
    v2 = {"1430": ("odira71@yahoo.com", "fodira"), "1457": ("lina@yahoo.com", "lakelo"),
          "20": ("achola@gmail.com ", "nope"), "21": (None, "ACHOLA"), "22": ("same@x.com", "dup")}
    user_of = oc.v2_user_lookup(v3, v2, "kisumu_v3")
    assert user_of("1430") == 3838            # uuid5 wins over the old placeholder account
    assert user_of(1457) == 2277              # placeholder
    assert user_of("20") == 4180 and user_of("21") == 4180   # e-mail, then username (case/space-insensitive)
    assert user_of("22") is None              # ambiguous → blank, never a guess
    assert user_of(None) is None and user_of("0") is None and user_of("999") is None


def test_inv_link_plan_pairs_by_shifted_time_and_never_overwrites():
    import uuid
    v2 = [{"id": 1, "visit": 100, "type": "laboratory", "price": 2370, "quantity": 1, "procedure": 27588,
           "user": 1530, "created_at": "2024-05-31 09:57:23"},
          {"id": 2, "visit": 100, "type": "laboratory", "price": 2370, "quantity": 1, "procedure": 27600,
           "user": 1530, "created_at": "2024-05-31 10:05:00"},
          {"id": 3, "visit": 101, "type": "Doctor", "price": 500, "quantity": 1, "procedure": 30,
           "user": 9, "created_at": "2024-06-01 08:00:00"},
          {"id": 4, "visit": 102, "type": "nursing", "price": 50, "quantity": 2, "procedure": 40,
           "user": 9, "created_at": "2024-06-02 08:00:00"}]
    v3 = [{"id": 10, "uuid": "a", "visit": 7, "type": "laboratory", "price": "2370.00", "quantity": 1,
           "procedure": None, "user": None, "created_at": "2024-05-31T06:57:23.000000Z"},   # V2 1, 3 h behind
          {"id": 11, "uuid": "b", "visit": 7, "type": "laboratory", "price": "2370.00", "quantity": 1,
           "procedure": None, "user": None, "created_at": "2024-05-31T12:00:00.000000Z"},   # time fits neither
          {"id": 12, "uuid": "c", "visit": 8, "type": "Doctor", "price": "500.00", "quantity": 1,
           "procedure": 999, "user": None, "created_at": "2024-06-01T05:00:00.000000Z"},    # already set
          {"id": 13, "uuid": str(uuid.uuid5(oc.USER_UUID_NS, "kisumu_v3/investigations/4")), "visit": 9,
           "type": "nursing", "price": "50.00", "quantity": 2, "procedure": None, "user": None,
           "created_at": None},                                                              # by uuid5
          {"id": 14, "uuid": "d", "visit": 55, "type": "x", "price": 1, "quantity": 1, "created_at": None}]
    plan, why = oc.inv_link_plan(v2, v3, facility="kisumu_v3", visit_of={100: 7, 101: 8, 102: 9}.get,
                                 proc_of=lambda v: v and v + 1000, user_of=lambda v: v and v + 1)
    plan = dict(plan)
    assert plan[10] == {"procedure": 28588, "user": 1531}
    assert plan[11] == {"user": 1531}               # V2 1 and 2 differ on procedure → left blank
    assert plan[12] == {"user": 10}                 # procedure 999 kept
    assert plan[13] == {"procedure": 1040, "user": 10}
    assert 14 not in plan and why["no V2 row (V3-only)"] == 1
    assert why["procedure: ambiguous (left blank)"] == 1
    assert why["procedure: set but V2 says otherwise (left)"] == 1
