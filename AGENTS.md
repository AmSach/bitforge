# BitForge repo notes

- BitForge is now the combined stack: weight compression, KV cache compression, context trimming, and block pruning.
- The important public APIs live in `bitforge/__init__.py` and should stay in sync with the README.
- `bitforge/kvcache.py` is the main bridge from the old KVQuant idea into this repo.
- `bitforge/prune.py` handles budget-based export shrinking.
- `bitforge/context.py` handles active-context trimming for fast small-device demos.
- Keep the docs honest: this is a small-device tool, not a magical 7B-on-ESP32 promise.
