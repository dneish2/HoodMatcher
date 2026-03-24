# Deployment Guide (Cloud Run + BigQuery)

This guide focuses on **secure, scalable defaults** for Neighborhood Matchmaker, with clear behavior for when BigQuery is used vs when CSV fallback is used.

## 1) Data source strategy

The app supports three source modes via `DATA_SOURCE_MODE`:

- `auto` (default): use BigQuery when `BIGQUERY_PROJECT`, `BIGQUERY_DATASET`, and `BIGQUERY_TABLE` are set; otherwise use local CSV if present.
- `bigquery`: require BigQuery. If config/query fails, no CSV fallback is used.
- `csv`: always read from local CSV path.

### Recommended production setting

Use:

```bash
DATA_SOURCE_MODE=bigquery
```

This prevents silent fallback to stale or incomplete local files in production.

### Recommended local/dev setting

Use:

```bash
DATA_SOURCE_MODE=auto
```

This keeps dev environments simple while still letting you test BigQuery when credentials/config are available.

---

## 2) BigQuery filtering behavior

City filtering is normalized (`St. Louis` -> `stlouis`) and controlled by `CITY_MATCH_MODE`:

- `contains` (default): most forgiving
- `exact`: strict equality on normalized city key
- `prefix`: starts-with behavior

For higher precision and lower scan/cost risk, prefer:

```bash
CITY_MATCH_MODE=exact
```

when your dataset's `Metro` naming is consistent.

---

## 3) Query efficiency

The loader does **not** use `SELECT *` on BigQuery. It always fetches:

- required columns: `RegionName`, `RegionType`, `Metro`
- optional date columns: from `BIGQUERY_DATE_COLUMNS`

Example:

```bash
BIGQUERY_DATE_COLUMNS=2024-01-31,2024-02-29,2024-03-31,2025-01-31,2025-02-28
```

This keeps bytes scanned predictable and reduces cost.

---

## 4) Caching model

The loader applies an in-process TTL cache keyed by normalized city.

- env var: `CACHE_TTL_SECONDS` (default: `600`)
- cache scope: per Cloud Run container instance

This helps repeated reads on warm instances but is **not a shared cache** across instances.

### Scaling note

For higher traffic and multi-instance consistency, adopt a shared cache (e.g., Redis/Memorystore or object cache). Keep the same city-key shape for compatibility.

---

## 5) Security and IAM best practices

### Cloud Run runtime identity

Use a dedicated service account for the service. Do not use user credentials in production.

### Minimum BigQuery permissions

Grant the runtime service account:

- `roles/bigquery.dataViewer` on the target dataset (or table-level equivalent)

Avoid broad project-wide roles when dataset-level access is enough.

### Credentials handling

- In Cloud Run, prefer Application Default Credentials (ADC) from the attached service account.
- Avoid mounting static JSON key files unless unavoidable.
- If a key file must be used (not recommended), store it in Secret Manager and rotate regularly.

### Auditability

Enable Cloud Audit Logs for BigQuery and monitor query patterns/cost.

---

## 6) Recommended production env var set

```bash
OPENAI_API_KEY=...
PROJECT_ID=...
LOCATION=us-central1

BIGQUERY_PROJECT=...
BIGQUERY_DATASET=...
BIGQUERY_TABLE=...
BIGQUERY_DATE_COLUMNS=2024-01-31,2024-02-29,2024-03-31,2025-01-31,2025-02-28

DATA_SOURCE_MODE=bigquery
CITY_MATCH_MODE=exact
CACHE_TTL_SECONDS=600
```

---

## 7) Operational checks

Confirm logs include structured loader events with:

- city
- selected source (`bigquery`, `csv`, `cache`, `none`)
- reason (e.g., `mode_forced`, `bigquery_config_missing`)
- row count
- cache hit status

This makes source decisions transparent and easy to debug.
