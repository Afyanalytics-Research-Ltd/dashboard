"""Support knowledge base for the HMIS support chatbot (RAG).

Sources live in support_kb/sources/. `python manage.py build_support_kb` chunks them,
embeds the chunks locally (fastembed) and upserts them into the pgvector database at
SUPPORT_KB_DB_URL. `support_kb.store.search()` is the hybrid retriever the n8n
support-chat workflow calls through /api/llm/v1/kb/search.
"""
