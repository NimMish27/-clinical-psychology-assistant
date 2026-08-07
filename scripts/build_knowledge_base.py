"""
build_knowledge_base.py
═══════════════════════
CLI entry point for the knowledge-base pipeline.

Usage:
    python scripts/build_knowledge_base.py build [--framework act] [--book <slug>]
    python scripts/build_knowledge_base.py index  [--force-embed]
    python scripts/build_knowledge_base.py all    [--framework act]

Stages:
    build   validate curated sources + manifests, chunk into
            knowledge/built/chunks.jsonl + parents.jsonl + hash_registry.json
    index   embed chunks and upsert into kb_{framework}_{index_version},
            then write knowledge/index/manifest.lock
    all     build then index
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("LOG_FORMAT", "console")

from app_logging.logger import get_logger, setup_logging  # noqa: E402

if TYPE_CHECKING:
    from knowledge_pipeline.models import BuildResult

_log = get_logger(__name__)


def cmd_build(args: argparse.Namespace) -> int:
    from knowledge_pipeline.builder import build_knowledge_base

    result = build_knowledge_base(framework=args.framework, book_slug=args.book)
    _print_result(result)
    return 1 if result.errors else 0


def cmd_index(args: argparse.Namespace) -> int:
    from knowledge_pipeline.indexer import index_chunks

    try:
        lock = index_chunks(force_embed=not args.incremental)
    except Exception as exc:
        _log.error("cli.index_failed", error=str(exc))
        print(f"\n[ERROR] index failed: {exc}", file=sys.stderr)
        return 1
    print("\n" + "=" * 60)
    print("  INDEX COMPLETE - manifest.lock")
    print("=" * 60)
    for framework, entry in sorted(lock.items()):
        print(f"  {framework:<12} -> {entry['collection']:<28} chunks={entry['chunk_count']}")
    print("=" * 60)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    build_code = cmd_build(args)
    if build_code != 0:
        return build_code
    return cmd_index(args)


def _print_result(result: BuildResult) -> None:
    print("\n" + "=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    print(f"  Books skipped:    {len(result.skipped)}")
    print(f"  Sources:          {len(result.source_files)}")
    print(f"  Parent units:     {result.parents}")
    print(f"  Child chunks:     {result.children}")
    print(f"  Total chars:      {result.total_chars}")
    print(f"  Total tokens:     {result.total_tokens}")
    if result.errors:
        print("  Errors:")
        for err in result.errors:
            print(f"    - {err}")
        print("  -> Build FAILED (see above)")
    else:
        print("  -> Build OK")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and index the clinical knowledge base.")
    sub = parser.add_subparsers(dest="stage", required=True)

    for name in ("build",):
        p = sub.add_parser(name, help="validate + chunk curated sources")
        p.add_argument("--framework", default=None, help="Restrict to one framework slug")
        p.add_argument("--book", default=None, help="Restrict to one book_slug")

    p_index = sub.add_parser("index", help="embed chunks into ChromaDB")
    p_index.add_argument(
        "--incremental",
        action="store_true",
        help="Only re-embed chunks whose content_hash changed",
    )

    p_all = sub.add_parser("all", help="build then index")
    p_all.add_argument("--framework", default=None, help="Restrict to one framework slug")
    p_all.add_argument("--book", default=None, help="Restrict to one book_slug")
    p_all.add_argument(
        "--incremental",
        action="store_true",
        help="Only re-embed chunks whose content_hash changed",
    )
    return parser


def main() -> None:
    setup_logging(level="INFO", log_format="console")
    args = build_parser().parse_args()
    dispatch = {
        "build": cmd_build,
        "index": cmd_index,
        "all": cmd_all,
    }
    sys.exit(dispatch[args.stage](args))


if __name__ == "__main__":
    main()
