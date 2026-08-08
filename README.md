# Oasis AdaMem

Adaptive Cognitive Memory — implementing and extending
[AdaMEM: Test-Time Adaptive Memory for Language Agents](https://arxiv.org/abs/2606.05684) (ICML 2026).

## Vision

> Current AI agents can remember information, but they do not truly learn from experience.
> Memory is not about storing the past. Memory is about changing the future.

See [IDEA.md](IDEA.md) for the full vision.

## Status

**M1 — Reproduction in progress** (see [plan](.hermes/plans/2026-08-08_213704-adamem-m1-reproduction.md)):

- Training-free inference pipeline on ALFWorld (seen/unseen), 5 memory mechanisms
- Environment: WSL2 Ubuntu-24.04 + DeepSeek API (policy) + local CPU embedding service
- Phase 0 done (env, data, embedding service, DeepSeek compat patch)

## Layout

```
reference/                  paper PDF + official code (yunx-z/AdaMEM, zero modification)
experiments/m1/             runbook, patches, smoke tests, logs
.hermes/plans/              milestone plans
```

## Quick Links

- [Runbook (reproduction recipe)](experiments/m1/RUNBOOK.md)
- [DeepSeek compat patch](experiments/m1/patches/deepseek-compat.patch)

## License

See [LICENSE](LICENSE).
