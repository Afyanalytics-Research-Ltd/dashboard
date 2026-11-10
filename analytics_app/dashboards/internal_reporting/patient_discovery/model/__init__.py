"""Warehouse model: measurements in, beliefs out.

concepts.py   what each concept means and which counting question it answers
measure.py    the aggregate measurements (SQL builders and parsers); counts and value shapes only
interpret.py  a pure function: catalogue + observations + statements -> beliefs, questions and measurement gaps
planner.py    the loop: interpret, measure the cheapest blocking gaps within a budget, interpret again

Nothing here writes to Snowflake or changes the counting refresh. The existing job (patient_discovery.job) is
unchanged and still produces today's statuses.
"""
