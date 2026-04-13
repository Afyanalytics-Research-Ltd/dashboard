import serpapi
import os
from dotenv import load_dotenv
load_dotenv()

# results = serpapi.GoogleSearch({
#     "engine"   : "google_trends",
#     "q"        : "COSRX",
#     "geo"      : "KE",
#     "date"     : "2023-01-01 2025-04-09",
#     "data_type": "TIMESERIES",
#     "api_key"  : os.getenv("SERPAPI_KEY"),
# }).get_dict()

# # Print top-level keys so we can see what came back
# print("Top level keys:", list(results.keys()))

# # Check interest_over_time structure
# if "interest_over_time" in results:
#     iot = results["interest_over_time"]
#     print("\ninterest_over_time keys:", list(iot.keys()))
#     print("\nFirst timeline point:")
#     print(iot["timeline_data"][0])
# else:
#     print("\nNo interest_over_time key found")
#     print("Full response:", results)


# import glob, os
# for f in glob.glob("embu_trends/checkpoints/*.csv"):
#     os.remove(f)
# for f in glob.glob("google_trends_output/checkpoints/*.csv"):
#     os.remove(f)
# print("All checkpoints cleared")


# import pandas as pd
# import numpy as np

# pp = pd.read_csv("tendri\\data\\pharmaplus_product_list.csv")
# pp["units_sold"] = pd.to_numeric(pp.get("units_sold"), errors="coerce").fillna(0)

# # What does units_sold look like?
# print("units_sold stats:")
# print(pp["units_sold"].describe())
# print(f"\nProducts with units_sold > 0: {(pp['units_sold'] > 0).sum()}")
# print(f"Products with units_sold = 0: {(pp['units_sold'] == 0).sum()}")

# # What does price look like?
# print("\nPrice stats:")
# print(pp["price_kes"].describe())

import pandas as pd
op = pd.read_csv("tendri\\data\\branch_106_opening_stock_products.csv")

# Check new categories
print("=== Categories in output_products ===")
print(op["Category"].value_counts().to_string())

print("\n=== New category rows ===")
new_cats = op[op["Category"].isin(["Beauty Products","Body Building"])]
print(new_cats.head(10).to_string())

print(f"\n=== New category count: {len(new_cats)} rows ===")