"""Build / refresh the support chatbot knowledge base.

    python manage.py build_support_kb            # incremental: only changed chunks are re-embedded
    python manage.py build_support_kb --rebuild  # re-embed everything
    python manage.py build_support_kb --dry-run  # chunk only, print stats

Sources: support_kb/sources/ (setup guide, screen reference, generated API reference).
Target: the pgvector database at SUPPORT_KB_DB_URL.
"""
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Chunk, embed (local fastembed) and upsert the support knowledge base into pgvector."

    def add_arguments(self, parser):
        parser.add_argument("--rebuild", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--batch", type=int, default=64)

    def handle(self, *args, rebuild=False, dry_run=False, batch=64, **kw):
        from collections import Counter

        from afya_backends import llm
        from support_kb import store
        from support_kb.chunker import load_all

        src = Path(settings.BASE_DIR) / "support_kb" / "sources"
        chunks = load_all(src)
        by_src = Counter(c.source for c in chunks)
        self.stdout.write(f"{len(chunks)} chunks: {dict(by_src)}; "
                          f"max {max(len(c.content) for c in chunks)} chars")
        if dry_run:
            return

        conn = store.connect()
        store.ensure_schema(conn)
        have = {} if rebuild else store.existing_hashes(conn)
        todo = [c for c in chunks if have.get(c.id) != c.content_hash]
        self.stdout.write(f"{len(todo)} chunks to embed ({len(chunks) - len(todo)} unchanged)")
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            vecs = llm.local_embed([c.content for c in part])
            store.upsert(conn, [(c.id, c.source, c.title, c.breadcrumb, c.url_path, c.content,
                                 c.content_hash, v) for c, v in zip(part, vecs)])
            self.stdout.write(f"  {min(i + batch, len(todo))}/{len(todo)}")
        removed = store.delete_missing(conn, {c.id for c in chunks})
        conn.close()
        self.stdout.write(self.style.SUCCESS(f"Support KB ready: {len(chunks)} chunks, {removed} stale removed"))
