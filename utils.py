import json
import pandas as pd


def load_events_table(session, table_name: str) -> pd.DataFrame:
    data = session.table(table_name)
    df = data.to_pandas()
    return df.set_index(["MODULE_SOURCE", "SOURCE_TABLE"])


def parse_payload_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["PAYLOAD"] = result["PAYLOAD"].apply(
        lambda x: json.loads(x) if pd.notna(x) else None
    )
    return result


def get_module_dataframe(df: pd.DataFrame, module: str) -> pd.DataFrame:
    return df.loc[module]

def get_modules(df: pd.DataFrame) -> pd.DataFrame:
    return df.index.get_level_values("MODULE_SOURCE").unique()

def get_module_tables(finance_df: pd.DataFrame) -> pd.Index:
    return finance_df.index.get_level_values("SOURCE_TABLE").unique()


def get_source_table(df: pd.DataFrame, module_source: str, source_table: str) -> pd.DataFrame:
    return df.loc[(module_source, source_table)]


def normalize_first_payload(df: pd.DataFrame) -> pd.DataFrame:
    payload = next(iter(df["PAYLOAD"]), None)
    return pd.json_normalize(payload)

def normalize_all_payloads(df: pd.DataFrame) -> pd.DataFrame:
    payloads = df["PAYLOAD"].dropna().tolist()
    if not payloads:
        return pd.DataFrame()
    return pd.json_normalize(payloads)


