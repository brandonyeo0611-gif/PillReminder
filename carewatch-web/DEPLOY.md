# CareWatch Deployment Guide

## API (Railway / Render / Fly.io)

```bash
uvicorn app.api:app --host 0.0.0.0 --port $PORT --workers 1
```

**Required:** `--workers 1` (SQLite does not handle concurrent writes)

**CORS:** Set on API host:
```
CORS_ORIGINS=https://your-app.vercel.app
```

## Frontend (Vercel)

- Build: `next build`
- Set env: `NEXT_PUBLIC_API_URL=https://your-api-host.railway.app`

## Smoke Test Before Wiring Frontend

```bash
curl https://your-api-host.railway.app/api/risk
```

Must return `{ "risk_score": ..., "risk_level": ... }` before deploying frontend.
