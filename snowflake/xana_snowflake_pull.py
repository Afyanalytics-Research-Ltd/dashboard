import snowflake.connector # type: ignore
import configparser
import os
import csv

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

sf = config["snowflake"]

totp = input("Enter your Duo TOTP code: ").strip()

conn = snowflake.connector.connect(
    account       = sf["account"],
    user          = sf["user"],
    warehouse     = sf["warehouse"],
    database      = sf["database"],
    schema        = sf["schema"],
    role          = sf["role"],
    password      = sf["password"],
    passcode      = totp,
    authenticator = sf["authenticator"]
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_FROM = "2025-09-01"

# Each entry: (csv_filename, query)
TABLES = [

    ("inventory_inventory_products", """
        SELECT DISTINCT
            f.value:id::INT              AS id,
            f.value:code::STRING         AS code,
            f.value:item_no::STRING      AS item_no,
            f.value:name::STRING         AS name,
            f.value:description::STRING  AS description,
            f.value:bar_code::STRING     AS bar_code,
            f.value:category::INT        AS category,
            f.value:department::INT      AS department,
            f.value:unit::INT            AS unit,
            f.value:tax_category::INT    AS tax_category,
            f.value:selling_price::FLOAT AS selling_price,
            f.value:active::INT           AS active,
            f.value:created_at::TIMESTAMP AS created_at,
            f.value:updated_at::TIMESTAMP AS updated_at,
            f.value:deleted_at::TIMESTAMP AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'inventory_inventory_products'
    """),

    ("inventory_inventory_categories", """
        SELECT DISTINCT
            f.value:id::INT                      AS id,
            f.value:code::STRING                 AS code,
            f.value:name::STRING                 AS name,
            f.value:parent::STRING               AS parent,
            f.value:parent_id::INT               AS parent_id,
            f.value:cash_markup::DECIMAL(10,2)   AS cash_markup,
            f.value:credit_markup::DECIMAL(10,2) AS credit_markup,
            f.value:created_at::TIMESTAMP        AS created_at,
            f.value:updated_at::TIMESTAMP        AS updated_at,
            f.value:deleted_at::TIMESTAMP        AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'inventory_inventory_categories'
        AND created_at::DATE >= '{DATE_FROM}'
    """),

    ("inventory_stores", """
        SELECT DISTINCT
            f.value:id::INT                                   AS id,
            f.value:code::STRING                              AS code,
            f.value:name::STRING                              AS name,
            f.value:description::STRING                       AS description,
            f.value:department_id::INT                        AS department_id,
            f.value:facility_id::INT                          AS facility_id,
            f.value:parent_store_id::INT                      AS parent_store_id,
            f.value:main_store::INT                           AS main_store,
            f.value:delivery_store::INT                       AS delivery_store,
            f.value:facility_store::INT                       AS facility_store,
            f.value:can_order_from_suppliers::INT             AS can_order_from_suppliers,
            f.value:can_update_product_prices::INT            AS can_update_product_prices,
            f.value:open_time::STRING                         AS open_time,
            f.value:close_time::STRING                        AS close_time,
            TRY_CAST(f.value:created_at::STRING AS TIMESTAMP) AS created_at,
            TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP) AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'inventory_stores'
    """),

    ("inventory_store_products", """
        SELECT DISTINCT
            f.value:id::INT                                             AS id,
            f.value:product_id::INT                                     AS product_id,
            f.value:store_id::INT                                       AS store_id,
            f.value:bar_code::STRING                                    AS bar_code,
            f.value:lot_no::STRING                                      AS lot_no,
            f.value:quantity::DECIMAL(18,2)                             AS quantity,
            f.value:re_order_level::STRING                              AS re_order_level,
            f.value:selling_price::STRING                               AS selling_price,
            f.value:discount_price::STRING                              AS discount_price,
            f.value:lower_limit_price::STRING                           AS lower_limit_price,
            f.value:lower_limit_percentage::STRING                      AS lower_limit_percentage,
            f.value:insurance_price::STRING                             AS insurance_price,
            f.value:total_insurance_price::STRING                       AS total_insurance_price,
            f.value:unit_cost::STRING                                   AS unit_cost,
            f.value:total_cost::STRING                                  AS total_cost,
            f.value:total_cash_price::STRING                            AS total_cash_price,
            f.value:bulk_pay_quantity::DECIMAL(18,2)                    AS bulk_pay_quantity,
            f.value:bulk_percentage_discount::STRING                    AS bulk_percentage_discount,
            CASE
                WHEN TRY_CAST(f.value:expiry_date::STRING AS DATE) < '1900-01-01' THEN NULL
                ELSE TRY_CAST(f.value:expiry_date::STRING AS DATE)
            END                                                          AS expiry_date,
            TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)           AS created_at,
            TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)           AS updated_at,
            TRY_CAST(f.value:deleted_at::STRING AS TIMESTAMP)           AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table LIKE 'inventory_store_products'
    """),

    ("inventory_inventory_stocks", """
        SELECT DISTINCT
            f.value:id::INT                                   AS id,
            f.value:product::INT                              AS product_id,
            f.value:quantity::DECIMAL(18,2)                   AS quantity,
            TRY_CAST(f.value:created_at::STRING AS TIMESTAMP) AS created_at,
            TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP) AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'inventory_inventory_stocks'
        ORDER BY product_id, created_at
    """),

    ("inventory_inventory_batch_product_sales", """
        SELECT DISTINCT
            f.value:id::INT                   AS id,
            f.value:receipt::STRING           AS receipt,
            f.value:visit_id::INT             AS visit_id,
            f.value:patient::INT              AS patient,
            f.value:customer::INT             AS customer,
            f.value:user::INT                 AS user,
            f.value:station_id::INT           AS station_id,
            f.value:facility_id::INT          AS facility_id,
            f.value:scheme_id::INT            AS scheme_id,
            f.value:payment_mode::STRING      AS payment_mode,
            f.value:status::STRING            AS status,
            f.value:amount::DECIMAL(10,2)     AS amount,
            f.value:shop::INT                 AS shop,
            f.value:insurance::STRING         AS insurance,
            f.value:canceled_by::STRING       AS canceled_by,
            f.value:created_at::TIMESTAMP_NTZ AS created_at,
            f.value:updated_at::TIMESTAMP_NTZ AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'inventory_inventory_batch_product_sales'
        AND created_at::DATE >= '{DATE_FROM}'
    """),

    ("evaluation_pos_sale_details", """
        SELECT DISTINCT
            f.value:id::INT                   AS id,
            f.value:sale_id::INT              AS sale_id,
            f.value:pos_id::INT               AS pos_id,
            f.value:item_id::INT              AS item_id,
            f.value:store_product_id::INT     AS store_product_id,
            f.value:service_id::INT           AS service_id,
            f.value:name::STRING              AS name,
            f.value:type::STRING              AS type,
            f.value:status::STRING            AS status,
            f.value:prescription_note::STRING AS prescription_note,
            f.value:quantity::DECIMAL(18,2)   AS quantity,
            f.value:price::DECIMAL(10,2)      AS price,
            f.value:amount::DECIMAL(10,2)     AS amount,
            f.value:discount::DECIMAL(10,2)   AS discount,
            f.value:round_up::DECIMAL(10,2)   AS round_up,
            f.value:created_at::TIMESTAMP_NTZ AS created_at,
            f.value:updated_at::TIMESTAMP_NTZ AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'evaluation_pos_sale_details'
        AND created_at::DATE >= '{DATE_FROM}'
    """),

    ("reception_patients", """
        SELECT
     DISTINCT
            f.value:id::INT                  AS id,
            f.value:patient_no::INT          AS patient_no,
            f.value:system_id::STRING        AS system_id,
            f.value:customer_id::INT         AS customer_id,
            f.value:first_name::STRING       AS first_name,
            f.value:middle_name::STRING      AS middle_name,
            f.value:last_name::STRING        AS last_name,
            f.value:dob::DATE                AS dob,
            f.value:dob_friendly::STRING     AS dob_friendly,
            f.value:age_friendly::STRING     AS age_friendly,
            f.value:sex::STRING              AS sex,
            f.value:email::STRING            AS email,
            f.value:secondary_email::STRING  AS secondary_email,
            f.value:mobile::STRING           AS mobile,
            f.value:telephone::STRING        AS telephone,
            f.value:alt_number::STRING       AS alt_number,
            f.value:id_no::STRING            AS id_no,
            f.value:kra_pin_number::STRING   AS kra_pin_number,
            f.value:source::STRING           AS source,
            f.value:status::INT              AS status,
            f.value:eligible_for_points::INT AS eligible_for_points,
            f.value:registered_by::INT       AS registered_by,
            f.value:employee_id::INT         AS employee_id,
            f.value:clinic_id::INT           AS clinic_id,
            f.value:other_details::STRING    AS other_details,
            f.value:created_at::TIMESTAMP    AS created_at,
            f.value:updated_at::TIMESTAMP    AS updated_at,
            f.value:deleted_at::TIMESTAMP    AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'reception_patients'
        AND created_at::DATE >= '{DATE_FROM}'
    """),

    ("points", """
        SELECT
            f.value:id::INT               AS id,
            f.value:customer_id::INT      AS customer_id,
            f.value:points::INT           AS points,
            f.value:type::STRING          AS type,
            f.value:note::STRING          AS note,
            f.value:created_at::TIMESTAMP AS created_at,
            f.value:updated_at::TIMESTAMP AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'points'
    """),

    ("users_roles", """
        SELECT
            r.value:id::INT                       AS role_id,
            r.value:name::STRING                  AS role_name,
            r.value:display_name::STRING          AS role_display_name,
            r.value:description::STRING           AS role_description,
            r.value:created_at::TIMESTAMP_NTZ     AS role_created_at,
            p.value:id::INT                       AS permission_id,
            p.value:name::STRING                  AS permission_name,
            p.value:display_name::STRING          AS permission_display_name,
            p.value:description::STRING           AS permission_description,
            p.value:module::STRING                AS permission_module,
            p.value:resource::STRING              AS permission_resource,
            p.value:special::INT                  AS permission_special,
            p.value:created_at::TIMESTAMP_NTZ     AS permission_created_at,
            p.value:updated_at::TIMESTAMP_NTZ     AS permission_updated_at,
            p.value:pivot:role_id::INT            AS pivot_role_id,
            p.value:pivot:permission_id::INT      AS pivot_permission_id
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) r,
        LATERAL FLATTEN(input => r.value:permissions) p
        WHERE source_table = 'users_roles'
        AND r.value:created_at::DATE  >= '{DATE_FROM}'
        AND p.value:created_at::DATE  >= '{DATE_FROM}'
    """),

    ("reception_reception_shifts", """
        SELECT
            f.value:id::INT                                AS id,
            f.value:user_id::INT                           AS user_id,
            f.value:facility_id::INT                       AS facility_id,
            f.value:shift_date::DATE                       AS shift_date,
            f.value:confirmed_by::STRING                   AS confirmed_by,
            f.value:confirmed_by_id::INT                   AS confirmed_by_id,
            f.value:close_confirmed_by::STRING             AS close_confirmed_by,
            f.value:close_confirmed_by_id::INT             AS close_confirmed_by_id,
            f.value:opening_balance::DECIMAL(10,2)         AS opening_balance,
            f.value:cashier_closing_balance::DECIMAL(10,2) AS cashier_closing_balance,
            f.value:system_closing_balance::DECIMAL(10,2)  AS system_closing_balance,
            f.value:closing_variance::DECIMAL(10,2)        AS closing_variance,
            f.value:total_sales::DECIMAL(10,2)             AS total_sales,
            f.value:total_cash::DECIMAL(10,2)              AS total_cash,
            f.value:total_card::DECIMAL(10,2)              AS total_card,
            f.value:total_mpesa::DECIMAL(10,2)             AS total_mpesa,
            f.value:total_pickups::DECIMAL(10,2)           AS total_pickups,
            f.value:opened_at::TIMESTAMP                   AS opened_at,
            f.value:closed_at::TIMESTAMP                   AS closed_at,
            f.value:created_at::TIMESTAMP                  AS created_at,
            f.value:updated_at::TIMESTAMP                  AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'reception_reception_shifts'
    """),

    ("finance_evaluation_payments", """
        SELECT DISTINCT
            f.value:id::INT                                AS id,
            f.value:receipt::STRING                        AS receipt,
            f.value:receipt_prefix::STRING                 AS receipt_prefix,
            f.value:patient::INT                           AS patient_id,
            f.value:patient_name::STRING                   AS patient_name,
            f.value:patient_no::STRING                     AS patient_no,
            f.value:user::INT                              AS user_id,
            f.value:user_name::STRING                      AS user_name,
            f.value:visit::STRING                          AS visit_id,
            f.value:sale::INT                              AS sale,
            f.value:sale_id::INT                           AS sale_id,
            f.value:invoice_id::INT                        AS invoice_id,
            f.value:facility_id::INT                       AS facility_id,
            f.value:scu_id::FLOAT                          AS scu_id,
            f.value:transaction_id::STRING                 AS transaction_id,
            f.value:status::STRING                         AS status,
            f.value:payment_mode::STRING                   AS payment_mode,
            f.value:etims_receipt_sign::STRING             AS etims_receipt_sign,
            f.value:discount_reason::STRING                AS discount_reason,
            f.value:amount::INT                            AS amount,
            f.value:total_amount_was::DECIMAL(18,2)        AS total_amount_was,
            f.value:total_cash_payment::DECIMAL(18,2)      AS total_cash_payment,
            f.value:payable_amount::DECIMAL(18,2)          AS payable_amount,
            f.value:cash_amount::DECIMAL(18,2)             AS cash_amount,
            f.value:card_amount::DECIMAL(18,2)             AS card_amount,
            f.value:mpesa_amount::DECIMAL(18,2)            AS mpesa_amount,
            f.value:jambopay_amount::DECIMAL(18,2)         AS jambopay_amount,
            f.value:pesa_pal_card_amount::DECIMAL(18,2)    AS pesa_pal_card_amount,
            f.value:pesa_pal_mpesa_amount::DECIMAL(18,2)   AS pesa_pal_mpesa_amount,
            f.value:cheque_amount::DECIMAL(18,2)           AS cheque_amount,
            f.value:giftcard_amount::DECIMAL(18,2)         AS giftcard_amount,
            f.value:loyalty_amount::DECIMAL(18,2)          AS loyalty_amount,
            f.value:points_amount::DECIMAL(18,2)           AS points_amount,
            f.value:voucher_amount::DECIMAL(18,2)          AS voucher_amount,
            f.value:waiver_amount::DECIMAL(18,2)           AS waiver_amount,
            f.value:discount::DECIMAL(18,2)                AS discount,
            f.value:deposit::DECIMAL(18,2)                 AS deposit,
            f.value:dispensing::DECIMAL(18,2)              AS dispensing,
            f.value:change::DECIMAL(18,2)                  AS change,
            f.value:vat_amount::DECIMAL(18,2)              AS vat_amount,
            f.value:patientAccount_amount::DECIMAL(18,2)   AS patient_account_amount,
            f.value:mpesa_fix_status::STRING               AS mpesa_fix_status,
            f.value:mpesa_fix_at::STRING                   AS mpesa_fix_at,
            f.value:mpesa_fix_confidence::STRING           AS mpesa_fix_confidence,
            f.value:mpesa_fix_old_txn_id::STRING           AS mpesa_fix_old_txn_id,
            f.value:mpesa_fix_suggested_txn_id::STRING     AS mpesa_fix_suggested_txn_id,
            TRY_CAST(f.value:receipt_date::STRING AS DATE)      AS receipt_date,
            TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)   AS created_at,
            TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)   AS updated_at,
            TRY_CAST(f.value:deleted_at::STRING AS TIMESTAMP)   AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'finance_evaluation_payments'
    """),

    ("finance_evaluation_payments_details", """
        SELECT
            f.value:id::INT                                    AS id,
            f.value:payment::INT                               AS payment_id,
            f.value:prescription_id::INT                       AS prescription_id,
            f.value:visit::INT                                 AS visit_id,
            f.value:store_id::INT                              AS store_id,
            f.value:consumable_id::INT                         AS consumable_id,
            f.value:investigation::INT                         AS investigation_id,
            f.value:deposit_id::INT                            AS deposit_id,
            f.value:visit_charge_id::INT                       AS visit_charge_id,
            f.value:visit_ward_id::INT                         AS visit_ward_id,
            f.value:patient_invoice::INT                       AS patient_invoice_id,
            f.value:item_name::STRING                          AS item_name,
            f.value:item_code::STRING                          AS item_code,
            f.value:item_classify::STRING                      AS item_classify,
            f.value:category::STRING                           AS category,
            f.value:tag::STRING                                AS tag,
            f.value:store_code::STRING                         AS store_code,
            f.value:ward_name::STRING                          AS ward_name,
            f.value:description::STRING                        AS description,
            f.value:quantity::DECIMAL(18,2)                    AS quantity,
            f.value:amount::DECIMAL(18,2)                      AS amount,
            f.value:price::DECIMAL(18,2)                       AS price,
            f.value:discount::DECIMAL(18,2)                    AS discount,
            f.value:vat_amount::DECIMAL(18,2)                  AS vat_amount,
            f.value:patient_invoice_amount::DECIMAL(18,2)      AS patient_invoice_amount,
            f.value:cost_deprecated::DECIMAL(18,2)             AS cost_deprecated,
            f.value:price_deprecated::DECIMAL(18,2)            AS price_deprecated,
            TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)  AS created_at,
            TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)  AS updated_at,
            TRY_CAST(f.value:deleted_at::STRING AS TIMESTAMP)  AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'finance_evaluation_payments_details'
    """),

    ("finance_payments_pesa_pal_card", """
        SELECT
            f.value:id::INT               AS id,
            f.value:invoice_id::INT       AS invoice_id,
            f.value:transaction_id::INT   AS transaction_id,
            f.value:number::STRING        AS number,
            f.value:payment::INT          AS payment,
            f.value:amount::INT           AS amount,
            f.value:created_at::TIMESTAMP AS created_at,
            f.value:updated_at::TIMESTAMP AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'finance_payments_pesa_pal_card'
    """),

    ("finance_vouchers", """
        SELECT
            f.value:id::INT                AS id,
            f.value:code::STRING           AS code,
            f.value:status::STRING         AS status,
            f.value:condition::STRING      AS condition,
            f.value:description::STRING    AS description,
            f.value:customer_id::INT       AS customer_id,
            f.value:balance::DECIMAL(10,2) AS balance,
            f.value:reward::DECIMAL(10,2)  AS reward,
            f.value:usage_limit::INT       AS usage_limit,
            f.value:times_used::INT        AS times_used,
            f.value:expiry_date::TIMESTAMP AS expiry_date,
            f.value:created_at::TIMESTAMP  AS created_at,
            f.value:updated_at::TIMESTAMP  AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'finance_vouchers'
    """),

    ("finance_invoices", """
        SELECT
            f.value:id::INT                               AS id,
            f.value:invoice_no::STRING                    AS invoice_no,
            f.value:invoice_no_prefix::STRING             AS invoice_no_prefix,
            f.value:invoice_date::DATE                    AS invoice_date,
            f.value:actual_invoice_creation_date::DATE    AS actual_invoice_creation_date,
            f.value:type_status::STRING                   AS type_status,
            f.value:status::INT                           AS status,
            f.value:source::STRING                        AS source,
            f.value:patient_id::INT                       AS patient_id,
            f.value:patient_no::STRING                    AS patient_no,
            f.value:patient_name::STRING                  AS patient_name,
            f.value:patient_signature::STRING             AS patient_signature,
            f.value:company_id::INT                       AS company_id,
            f.value:corporate_id::INT                     AS corporate_id,
            f.value:scheme_id::INT                        AS scheme_id,
            f.value:policy_no::STRING                     AS policy_no,
            f.value:claim_no::STRING                      AS claim_no,
            f.value:independent_payer::STRING             AS independent_payer,
            f.value:amount::DECIMAL(10,2)                 AS amount,
            f.value:balance::DECIMAL(10,2)                AS balance,
            f.value:payable_amount::DECIMAL(10,2)         AS payable_amount,
            f.value:credit_amount::DECIMAL(10,2)          AS credit_amount,
            f.value:exemption_amount::DECIMAL(10,2)       AS exemption_amount,
            f.value:paid::INT                             AS paid,
            f.value:for_cash::INT                         AS for_cash,
            f.value:credit_note_number::STRING            AS credit_note_number,
            f.value:credit_note_reason::STRING            AS credit_note_reason,
            f.value:credit_note_status::STRING            AS credit_note_status,
            f.value:split_id::INT                         AS split_id,
            f.value:split_bill_id::INT                    AS split_bill_id,
            f.value:batch_sale_id::INT                    AS batch_sale_id,
            f.value:user_id::INT                          AS user_id,
            f.value:user_name::STRING                     AS user_name,
            f.value:staff_id::INT                         AS staff_id,
            f.value:unbilled_by::INT                      AS unbilled_by,
            f.value:is_interim::INT                       AS is_interim,
            f.value:is_manual::INT                        AS is_manual,
            f.value:auto_cancelled::INT                   AS auto_cancelled,
            f.value:etims_sign::STRING                    AS etims_sign,
            f.value:scu_id::STRING                        AS scu_id,
            f.value:store_code::STRING                    AS store_code,
            f.value:visit::INT                            AS visit,
            f.value:notes::STRING                         AS notes,
            f.value:comments::STRING                      AS comments,
            f.value:created_at::TIMESTAMP                 AS created_at,
            f.value:updated_at::TIMESTAMP                 AS updated_at,
            f.value:deleted_at::TIMESTAMP                 AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'finance_invoices'
    """),

    ("finance_invoice_payments", """
        SELECT
            f.value:id::INT                AS id,
            f.value:invoice_id::INT        AS invoice_id,
            f.value:receipt_no::STRING     AS receipt_no,
            f.value:amount::DECIMAL(10,2)  AS amount,
            f.value:by_insurance::STRING   AS by_insurance,
            f.value:company_id::INT        AS company_id,
            f.value:scheme_id::INT         AS scheme_id,
            f.value:batch::STRING          AS batch,
            f.value:user_id::INT           AS user_id,
            f.value:created_at::TIMESTAMP  AS created_at,
            f.value:updated_at::TIMESTAMP  AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'finance_invoice_payments'
    """),

    ("finance_invoice_items", """
        SELECT
            f.value:id::INT                                    AS id,
            f.value:invoice_id::INT                            AS invoice_id,
            f.value:item_id::INT                               AS item_id,
            f.value:split_bill_id::INT                         AS split_bill_id,
            f.value:store_id::INT                              AS store_id,
            f.value:master::INT                                AS master,
            f.value:item_name::STRING                          AS item_name,
            f.value:item_type::STRING                          AS item_type,
            f.value:item_classify::STRING                      AS item_classify,
            f.value:item_code::STRING                          AS item_code,
            f.value:category::STRING                           AS category,
            f.value:tag::STRING                                AS tag,
            f.value:revenue_summary_tag::STRING                AS revenue_summary_tag,
            f.value:store_code::STRING                         AS store_code,
            f.value:ward_name::STRING                          AS ward_name,
            f.value:amount::DECIMAL(18,2)                      AS amount,
            f.value:price::DECIMAL(18,2)                       AS price,
            f.value:quantity::DECIMAL(18,2)                    AS quantity,
            f.value:units::DECIMAL(18,2)                       AS units,
            TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)  AS created_at,
            TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)  AS updated_at,
            TRY_CAST(f.value:deleted_at::STRING AS TIMESTAMP)  AS deleted_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'finance_invoice_items'
        AND created_at::DATE >= '{DATE_FROM}'
    """),

    ("evaluation_prescription_payments", """
        SELECT
            f.value:id::INT                   AS id,
            f.value:prescription_id::INT      AS prescription_id,
            f.value:split_bill_id::INT        AS split_bill_id,
            f.value:amount::DECIMAL(10,2)     AS amount,
            f.value:price::DECIMAL(10,2)      AS price,
            f.value:cost::DECIMAL(10,2)       AS cost,
            f.value:discount::DECIMAL(10,2)   AS discount,
            f.value:quantity::DECIMAL(18,2)   AS quantity,
            f.value:dispensing_quantity::DECIMAL(18,2) AS dispensing_quantity,
            f.value:complete::INT             AS complete,
            f.value:paid::INT                 AS paid,
            f.value:invoiced::INT             AS invoiced,
            f.value:transfer::STRING          AS transfer,
            f.value:changes_track::STRING     AS changes_track,
            f.value:created_at::TIMESTAMP_NTZ AS created_at,
            f.value:updated_at::TIMESTAMP_NTZ AS updated_at
        FROM EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = 'evaluation_prescription_payments'
        AND created_at::DATE >= '{DATE_FROM}'
    """),
]

try:
    cs = conn.cursor()

    for table_name, query in TABLES:
        filename = os.path.join(OUTPUT_DIR, f"{table_name}.csv")
        if os.path.exists(filename):
            print(f"\n[{table_name}] Skipping — file already exists.")
            continue

        print(f"\n[{table_name}] Fetching...")

        cs.execute(query.replace("{DATE_FROM}", DATE_FROM))
        rows = cs.fetchall()
        columns = [desc[0] for desc in cs.description]

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            writer.writerows(rows)

        print(f"  Saved: {filename}  ({len(rows)} rows)")

    print(f"\nDone. All CSVs saved to: {OUTPUT_DIR}")

finally:
    cs.close()
    conn.close()
