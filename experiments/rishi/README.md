# TRACER — Finals Ranker

Trap-aware Retrieval And Career Evidence Ranker for the Redrob Senior AI Engineer JD.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for full technical design, honeypot strategy, and interview talking points.

## Quick start

```bash
# Place candidates.jsonl locally (not in repo)
# Copy or generate artifacts/ (see ARCHITECTURE.md)

python rank.py --candidates /path/to/candidates.jsonl --out submission.csv --artifacts artifacts
python scripts/submit_guard.py --submission submission.csv --candidates /path/to/candidates.jsonl
```

## Status

- Rank: ~32s on 100K CPU
- 0 honeypots in top-100
- 94/100 STRONG+GOOD on verification

## Optional GPU steps

Multi-query facets (`precompute_mq.py`) — **recommended**, ~8 min GPU.  
Cross-encoder rerank (`precompute_rerank.py`) — **optional**, skip if download/GPU issues.
