# Cortex v2 Re-Import — Post-Restart Verification Instructions

## What happened before this session

The Cortex v2 import pipeline was broken. A retrieval quality audit found:
- All 329 docs had 2026-04-06 timestamps (import time, not original dates)
- Zero inter-object relationships (graph score always 0)
- Summaries were filename slugs, not real text
- ~165 duplicate documents (original + re-imported copies)
- confidence=0 and tier=archive on everything

### What was fixed (11 files, 559 insertions, 549 tests passing)

**Files modified in ~/Lab/cortex:**
- `src/cortex/db/content_store.py` — Optional `created_at`/`updated_at` params on `insert()`
- `src/cortex/db/graph_store.py` — Optional `captured_at` param on `create_object()`
- `src/cortex/db/store.py` — Timestamps threaded through `create()` to both stores
- `src/cortex/pipeline/orchestrator.py` — Timestamps threaded through `capture()`
- `src/cortex/pipeline/importer.py` — **Major rewrite**: block scalar frontmatter parser, skip `source: ingest:` files, skip `type: index`, content-only dedup hash, preserve timestamps/summary/confidence from frontmatter, extract wiki-links, route through pipeline, second-pass wiki-link relationship resolution, entity extraction from tags+key_topics
- `src/cortex/cli/main.py` — `import-vault` wired to pipeline, `--skip-pipeline` flag added
- Test files: 17 new tests across 5 test files

### What should have been run between sessions

In a regular terminal (NOT Claude Code — Oxigraph needs exclusive access):
```bash
cd ~/Lab/cortex
rm -f ~/.cortex/cortex.db ~/.cortex/cortex.db-shm ~/.cortex/cortex.db-wal ~/.cortex/graph.db
.venv/bin/cortex init
.venv/bin/cortex import-vault ~/path/to/your/vault/
```

Expected output from import:
- `Imported: ~160`
- `Skipped: ~193 (duplicates/filtered)`
- `Failed: 0`
- `Wiki-link relationships: ~233`
- Per-doc pipeline messages (normalize/link/enrich/reason)
- Total time: ~3-5 minutes (embedding generation ~1s/doc)

---

## Verification Steps (run these in order)

### Step 1: Confirm MCP connection
Fetch cortex tool schemas and run `cortex_status`.

**Expected:**
- MCP tools resolve (cortex server connected)
- ~160 documents (NOT 329 — that's the old broken import)
- Entities > 0 (extracted from tags + key_topics)
- Graph triples > 304 (304 = ontology only; more means objects are in the graph)

**If 329 docs:** The re-import didn't run. User needs to exit Claude Code and run the terminal commands above.
**If 0 docs:** Stores were wiped but import didn't run.
**If triples = 304:** Import ran with `--skip-pipeline` or Oxigraph was locked. Re-run without Claude Code.

### Step 2: Verify timestamps are original dates (not import time)
Run `cortex_search` query="database locked" doc_type="fix" limit=3.

**Check:** The `created_at` field should start with `2026-03-23` (when the fix was written), NOT `2026-04-06` (old import date). There should be only ONE result for this fix (not two duplicates).

### Step 3: Verify summaries are real text
Run `cortex_context` topic="Artemis architecture decisions" limit=5.

**Check:** Each result's `summary` field should be human-readable text like "Artemis tech stack decisions: React+TypeScript frontend, FastAPI backend..." — NOT a filename slug like "2026-03-20-artemis-architecture-tech-stack-decisions".

### Step 4: Verify no duplicates
Run `cortex_list` doc_type="fix" limit=50.

**Check:** ~40 fixes total. NO duplicate pairs with same content but different date prefixes (the old import had pairs like `2026-03-23-sqlite-database-locked...` AND `2026-04-04-sqlite-database-locked...` for the same fix).

### Step 5: Verify entities exist in graph
Run `cortex_graph` entity="sqlite".

**Check:** The `objects` list should have multiple documents mentioning SQLite. The `connections` list should NOT be empty (wiki-link relationships should appear as `supports` edges).

### Step 6: Verify inter-object relationships
Pick a fix document ID from Step 2. Run `cortex_graph` obj_id=that_id.

**Check:** The `relationships` list should NOT be empty. Wiki-links from `## Related` sections were converted to `supports` relationships during import.

### Step 7: Verify graph boost in search scoring
Run `cortex_search` query="SQLite concurrent writes" limit=5.

**Check:** In `score_breakdown`, the `graph` component should be > 0 for well-connected documents. In the old import, graph was ALWAYS 0.

### Step 8: Verify dossier intelligence works
Run `cortex_dossier` topic="SQLite".

**Expected structure:**
- `entity`: found SQLite entity with type "technology"
- `objects`: list of fixes, sessions, research mentioning SQLite
- `related_entities`: connected concepts (concurrent-writes, database-lock, etc.)
- `timeline`: chronologically ordered (oldest to newest, spanning real dates)
- `contradictions`: may be empty (OK)

### Step 9: Verify recency scoring uses real dates
Run `cortex_search` query="knowledge base" limit=10.

**Check:** The `created_at` values across results should span different dates (2026-03-19 through 2026-04-06). In the old import, ALL docs had recency ~0.99 because they all had the same import-time timestamp.

### Step 10: Run the automated test suite
```bash
cd /Users/fabrizzio/Lab/cortex && .venv/bin/python -m pytest tests/ --tb=short -q
```

**Expected:** 549 passed, 0 failed. (532 original + 17 new tests for the import pipeline fixes.)

---

## After all 10 steps pass

### Capture a validation record
Use `cortex_capture` with:
- title: "Cortex v2 import pipeline fix validated"
- obj_type: "lesson"
- content: "Full vault re-import with fixed pipeline verified. Results: ~160 unique docs imported (193 filtered: 169 source:ingest re-imports + 24 index/dedup). Original timestamps preserved from frontmatter. Real summaries stored (109/160 have summaries). 233 wiki-link relationships created via second-pass fuzzy title matching. Entities extracted from tags + key_topics. Pipeline runs normalize→link→enrich→reason on each doc. 549 tests passing."
- summary: "Import pipeline fix validated: timestamps, summaries, dedup, wiki-links, entities all working correctly after full re-import."
- entities: [{"name": "Cortex", "type": "project"}, {"name": "Oxigraph", "type": "technology"}, {"name": "SQLite", "type": "technology"}, {"name": "MCP", "type": "technology"}]
- tags: "cortex,import-pipeline,validation,fix"
- project: "cortex"

### Update memory
Update the file `/Users/fabrizzio/.claude/projects/-Users-fabrizzio-Lab/memory/project_cortex_status.md` with:

```markdown
---
name: cortex_v2_current_status
description: Cortex v2 fully operational with clean re-imported data (2026-04-06)
type: project
---

Cortex v2 MCP is live, all tools operational, data quality verified after import pipeline fix.

**Status:** 549 tests. ~160 docs, entities populated, wiki-link relationships active.
**Data:** ~160 unique docs from Obsidian vault, real timestamps (2026-03-19 to 2026-04-06), 109 with summaries, ~233 wiki-link relationships, entities from tags+key_topics.
**Repo:** ~/Lab/cortex — https://github.com/abbacusgroup/Cortex.git

## Import pipeline (fixed 2026-04-06)

11 files changed across store layer, pipeline, importer, and CLI:
- Timestamps preserved from YAML frontmatter (created/updated fields)
- Summaries preserved from frontmatter (including >- block scalar syntax)
- Duplicates filtered: skip `source: ingest:` re-imports + `type: index` scaffolding + content-only dedup hash
- Full pipeline runs on import: normalize (embeddings) → link (entities + relationships) → enrich (tier) → reason (OWL-RL inference)
- Wiki-links parsed from ## Related sections → `supports` relationships via fuzzy title matching
- Entities extracted from tags + key_topics frontmatter fields

## Known limitations
- Dashboard and CLI/MCP can't run simultaneously (Oxigraph single-writer lock)
- litellm optional — LLM relationship discovery degrades gracefully when unavailable
- 51/160 docs have confidence=0 (no summary in frontmatter → can't auto-classify without LLM)
- Wiki-link resolution uses 0.8 similarity threshold — some links may not resolve
```

### Update MEMORY.md reference line
Change the Cortex status line in MEMORY.md from:
`- [project_cortex_status.md](project_cortex_status.md) — Cortex v2 MCP registered but tools not yet connecting (2026-04-06), needs restart`
to:
`- [project_cortex_status.md](project_cortex_status.md) — Cortex v2 fully operational, clean re-import with fixed pipeline (2026-04-06)`

---

## Key file locations for reference

| What | Path |
|------|------|
| Cortex v2 source | `~/Lab/cortex/src/cortex/` |
| Test suite | `~/Lab/cortex/tests/` (549 tests) |
| Python venv | `~/Lab/cortex/.venv/` |
| Data directory | `~/.cortex/` (cortex.db + graph.db) |
| Ontology | `~/Lab/cortex/ontology/cortex.ttl` |
| Obsidian vault | `~/path/to/your/vault/` (source data) |
| MCP config | `~/.claude.json` (mcpServers.cortex) |
| MCP command | `/Users/fabrizzio/Lab/cortex/.venv/bin/cortex serve --transport stdio` |
| Build plan | `~/Lab/cortex/` — see cortex_search "Cortex v2 build plan" |
| Import pipeline | `src/cortex/pipeline/importer.py` (ObsidianImporter) |
| Pipeline orchestrator | `src/cortex/pipeline/orchestrator.py` |
| CLI entry | `src/cortex/cli/main.py` |

## Architecture quick reference

- **Dual store**: Oxigraph (RDF graph, SPARQL) + SQLite (content, FTS5, embeddings)
- **8 knowledge types**: decision, lesson, fix, session, research, source, synthesis, idea
- **8 relationship types**: causedBy, ledTo, contradicts, supports, supersedes, dependsOn, implements, mentions
- **Pipeline stages**: normalize → link → enrich → reason
- **17 MCP tools**: 11 public + 6 admin
- **OWL-RL inference**: symmetric (contradicts), transitive (supersedes), inverse (causedBy↔ledTo)
