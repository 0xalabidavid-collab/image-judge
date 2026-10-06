FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY supabase ./supabase

# Uploaded images are cached here; the source of truth is the Supabase bucket, so this can be wiped.
ENV IMAGE_JUDGE_UPLOADS=/tmp/uploads

# One process only: benchmark runs are background jobs held in memory.
# --proxy-headers lets the app see that the browser used https behind Railway's proxy (secure cookies).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
