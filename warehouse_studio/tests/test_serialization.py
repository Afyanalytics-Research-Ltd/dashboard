"""serialization: cursor output -> JSON-safe values and column metadata."""

import datetime as dt
import json
import uuid
from collections import namedtuple
from decimal import Decimal

from django.test import SimpleTestCase

from warehouse_studio.services import serialization as ser

Meta = namedtuple('ResultMetadata', 'name type_code display_size internal_size precision scale is_nullable')


class ValueTests(SimpleTestCase):
    def test_scalars(self):
        self.assertIsNone(ser.to_json_value(None))
        self.assertIs(ser.to_json_value(True), True)
        self.assertEqual(ser.to_json_value(5), 5)
        self.assertEqual(ser.to_json_value(1.5), 1.5)
        self.assertEqual(ser.to_json_value('x'), 'x')

    def test_big_int_as_string(self):
        self.assertEqual(ser.to_json_value(2 ** 53 - 1), 2 ** 53 - 1)
        self.assertEqual(ser.to_json_value(2 ** 53), str(2 ** 53))
        self.assertEqual(ser.to_json_value(-(2 ** 60)), str(-(2 ** 60)))

    def test_float_specials(self):
        self.assertEqual(ser.to_json_value(float('nan')), 'nan')
        self.assertEqual(ser.to_json_value(float('inf')), 'inf')

    def test_decimal(self):
        self.assertEqual(ser.to_json_value(Decimal('42')), 42)
        self.assertIsInstance(ser.to_json_value(Decimal('42.000')), int)
        self.assertEqual(ser.to_json_value(Decimal('1.5')), 1.5)
        self.assertEqual(ser.to_json_value(Decimal('0.1')), 0.1)
        self.assertEqual(ser.to_json_value(Decimal('0.12345678901234567890')), '0.12345678901234567890')
        self.assertEqual(ser.to_json_value(Decimal('NaN')), 'NaN')
        self.assertEqual(ser.to_json_value(Decimal('123456789012345678901234')), '123456789012345678901234')

    def test_temporal(self):
        tz = dt.timezone(dt.timedelta(hours=3))
        self.assertEqual(ser.to_json_value(dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=tz)), '2026-01-02T03:04:05+03:00')
        self.assertEqual(ser.to_json_value(dt.date(2026, 1, 2)), '2026-01-02')
        self.assertEqual(ser.to_json_value(dt.time(3, 4, 5)), '03:04:05')
        self.assertEqual(ser.to_json_value(dt.timedelta(minutes=1, seconds=30)), 90.0)

    def test_binary_uuid(self):
        self.assertEqual(ser.to_json_value(b'\x01\xab'), {'$binary': '01AB'})
        self.assertEqual(ser.to_json_value(bytearray(b'\xff')), {'$binary': 'FF'})
        self.assertEqual(len(ser.to_json_value(b'\x00' * 5000)['$binary']), 4096)
        u = uuid.uuid4()
        self.assertEqual(ser.to_json_value(u), str(u))

    def test_json_family(self):
        self.assertEqual(ser.to_json_value('{"a": [1, 2]}', 'json'), {'$json': {'a': [1, 2]}})
        self.assertEqual(ser.to_json_value('not json', 'json'), 'not json')
        self.assertEqual(ser.to_json_value('{"a": 1}', 'text'), '{"a": 1}')

    def test_containers_and_fallback(self):
        self.assertEqual(ser.to_json_value([Decimal('1'), None]), [1, None])
        self.assertEqual(ser.to_json_value({1: Decimal('2.5')}), {'1': 2.5})
        self.assertEqual(ser.to_json_value(object.__new__(type('X', (), {'__str__': lambda s: 'x!'}))), 'x!')

    def test_everything_is_json_dumpable(self):
        row = [Decimal('1.25'), dt.datetime(2026, 1, 1), b'\x00', 2 ** 70, float('nan'), '[1]']
        cols = [{'family': f} for f in ('number', 'timestamp', 'binary', 'number', 'number', 'json')]
        json.dumps(ser.rows_to_json([row], cols))

    def test_plain(self):
        self.assertEqual(ser.plain('{"a": 1}'), {'a': 1})
        self.assertEqual(ser.plain('hello'), 'hello')
        self.assertEqual(ser.plain(Decimal('3')), 3)
        self.assertIsNone(ser.plain(None))


class ColumnTests(SimpleTestCase):
    def test_result_metadata(self):
        cols = ser.describe_columns([Meta('ID', 0, None, None, 38, 0, False), Meta('AMT', 0, None, None, 12, 2, True),
                                     Meta('V', 5, None, None, None, None, True), Meta('TS', 6, None, None, 0, 9, True),
                                     Meta('B', 13, None, None, None, None, True)])
        self.assertEqual([c['label'] for c in cols], ['NUMBER(38,0)', 'NUMBER(12,2)', 'VARIANT', 'TIMESTAMP_LTZ', 'BOOLEAN'])
        self.assertEqual([c['family'] for c in cols], ['number', 'number', 'json', 'timestamp', 'boolean'])
        self.assertFalse(cols[0]['nullable'])
        self.assertEqual(cols[1]['scale'], 2)

    def test_plain_tuples(self):
        cols = ser.describe_columns([('A', 2, None, None, None, None, True), ('D', 3), ('T', 12, None, None, None, None)])
        self.assertEqual([c['type'] for c in cols], ['TEXT', 'DATE', 'TIME'])
        self.assertEqual([c['family'] for c in cols], ['text', 'date', 'time'])
        self.assertTrue(cols[1]['nullable'])

    def test_duplicate_names(self):
        cols = ser.describe_columns([('ID', 0), ('ID', 0), ('ID', 0), ('X', 0)])
        self.assertEqual([c['key'] for c in cols], ['ID', 'ID_1', 'ID_2', 'X'])
        self.assertEqual([c['name'] for c in cols], ['ID', 'ID', 'ID', 'X'])

    def test_type_helpers(self):
        self.assertEqual(ser.type_name('number'), 'NUMBER')
        self.assertEqual(ser.type_name(999), 'TEXT')
        self.assertEqual(ser.type_family('BINARY'), 'binary')
        self.assertEqual(ser.type_family('ARRAY'), 'json')
        self.assertEqual(ser.type_family('REAL'), 'number')
        self.assertEqual(ser.describe_columns(None), [])
        self.assertEqual(ser.describe_columns([('N', 0, None, None, None, None, True)])[0]['label'], 'NUMBER')

    def test_records(self):
        recs = ser.records([(Decimal('1'), '{"k": true}')], [('ID', 0), ('OBJ', 9)])
        self.assertEqual(recs, [{'id': 1, 'obj': {'k': True}}])
