from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from warehouse.services.snowflake import SnowflakeClient
import pandas as pd

def snowflake_query_view(request):

    df = None
    error = None
    query = ""

    if request.method == "POST":
        query = request.POST.get("query")

        try:
            sf = SnowflakeClient()

            # ⚠️ basic safety guard (optional but recommended)
            if "drop" in query.lower() or "delete" in query.lower():
                raise Exception("Dangerous query not allowed")

            df = sf.query(query)

        except Exception as e:
            error = str(e)

    return render(request, "snowflake_query.html", {
        "df": df.to_html(classes="table table-striped") if df is not None else None,
        "error": error,
        "query": query
    })