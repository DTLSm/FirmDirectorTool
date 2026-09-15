# FirmDirectorTool

Board interlock distances and appointment prediction, built entirely on public SEC EDGAR data.

> **Status:** Slice 0 — distance engine and CI. Nothing is deployed yet. See [the roadmap](#roadmap).

## What it will do

1. **Ingest** SEC Section 16 filings (Forms 3, 4, 5) daily, idempotently, with catch-up.
2. **Build** the director–firm interlock graph from persistent reporting-owner CIKs.
3. **Compute** Mahalanobis distances between directors on the same board, over a hybrid
   feature vector of attributes derivable from filings plus network structure.
4. **Predict** future board appointments with a link-prediction model, versioned and monitored.
5. **Serve** all of it through an API and a small web front end, on AKS, provisioned with Terraform.

## Why EDGAR, and what's deliberately left out

Section 16 filings are US government works in the public domain, so the entire pipeline — code
*and* data — can be published. The `rptOwnerCik` on each filing is a persistent identifier for the
individual, which sidesteps the name-disambiguation problem that dominates interlock research.

Two things are deliberately **not** in the feature set:

- **Demographics and career history** (age, gender, race, education, prior non-executive roles).
  They are not derivable from structured public filings. Inferring protected attributes from names
  is something a public tool should not do.
- **Board exit dates.** There is no Section 16 filing for leaving a board; people simply stop
  filing. Tenure is therefore right-censored and inferred by a documented rule, validated against
  a sample of DEF 14A proxy statements. See `docs/data-model.md` once it exists.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups          # create .venv and install everything
uv run pre-commit install     # ruff on every commit
uv run pytest                 # tests
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

CI runs the same four commands on every push and pull request.

## The distance engine

`src/firmdirectortool/distance/` — ported from a private BoardEx-based research tool, with three
changes: DataFrame in / DataFrame out, Mahalanobis only, and ambiguous focal directors raise
instead of being silently resolved.

The whitening transform (`WhiteningTransform`) is a versioned artifact. It is fitted once on a
reference population and applied consistently, so distances computed at different times remain
comparable. It uses the pseudo-inverse of the covariance, so collinear or constant features are
dropped rather than inverted. Rationale in `docs/decisions/0001-whitening-transform-as-artifact.md`.

## Roadmap

| Slice | Deliverable |
|---|---|
| **0** | Repo, CI, distance engine with tests — *this* |
| 1 | EDGAR client, parsers, Postgres schema, graph, features, FastAPI — all local via docker-compose |
| 2 | Azure: Terraform (persistent data stack + ephemeral compute stack), OIDC federation, ACR, AKS |
| 3 | Kubernetes: manual deploy, CronJob ingestion, debugging drills |
| 4 | GitOps with Argo CD |
| 5 | Link prediction, MLflow registry, training as Jobs |
| 6 | Prometheus/Grafana, data and model drift, retrospective precision@k |
| 7 | Web front end, ADRs, demo recording, timed rebuild |

## Licence

MIT — see `LICENSE`.
