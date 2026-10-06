# Running Image Judge on Supabase

With three settings in `.env`, the app saves everything (tasks, answers, reasons, lessons, run history,
judge cache) to Postgres and mirrors every image to a private Storage bucket. Without them it keeps
using the local SQLite file, exactly as before.

## 1. Create the Supabase project

1. Create a project at supabase.com (note the database password you choose).
2. **Settings > Database > Connection string**: copy the **Session pooler** URI
   (`postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`).
3. **Settings > API Keys**: copy the **service_role** secret key. It bypasses all security: never put it
   in a web page, a chat, or a repository.

## 2. Add the settings to `.env`

```
IMAGE_JUDGE_DATABASE_URL=postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_SERVICE_KEY=<service_role key>
```

(`SUPABASE_BUCKET` defaults to `image-judge`; the migration creates it as a private bucket.)

## 2b. Recommended: keep images on Cloudflare R2 instead (10 GB free)

Supabase's free plan has 1 GB of file storage and your images are about 1 GB. Cloudflare R2 includes 10 GB
and free downloads, so use it for images and keep Supabase for the database and logins.
(Backblaze B2 also gives 10 GB free and works the same way.)

1. Cloudflare dashboard > **R2 Object Storage** > **Create bucket** named `image-judge` (leave it private;
   do not turn on public access). R2 needs a payment method on file, but nothing is charged under the free limits.
2. **R2 > Manage API Tokens > Create API token**: permission **Object Read & Write**, limited to that bucket.
   Copy the **Access Key ID**, **Secret Access Key**, and the account's S3 endpoint
   (`https://<account-id>.r2.cloudflarestorage.com`).
3. Add to `.env`:

```
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
S3_ACCESS_KEY_ID=<access key id>
S3_SECRET_ACCESS_KEY=<secret access key>
S3_BUCKET=image-judge
S3_REGION=auto
```

For Backblaze B2 use its S3 endpoint (`https://s3.<region>.backblazeb2.com`), an application key limited to the
bucket, and `S3_REGION=<region>` (for example `us-west-004`).

When the S3 settings are present they are used for images; `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` are then
only needed for hosting login (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) and no longer for storage.

## 3. Copy your existing data

```
python scripts/migrate_to_supabase.py            # dry run: counts and total image size
python scripts/migrate_to_supabase.py --apply    # copy, then verify row counts and image sizes
```

Safe to repeat. Your local database and images are left untouched as a backup.

## Security

- Every table has row-level security on and no policies, so the public Supabase API (anon key) can read and
  write nothing. Only the connection string and service key can. Keep both secret.
- The bucket is private; the app serves images itself.
- Image files cost storage: Supabase's free plan includes 1 GB (Pro 100 GB); Cloudflare R2 and Backblaze B2
  include 10 GB free.

## Getting the data out for training

Everything is plain tables with JSON columns, so it can be queried in the Supabase SQL editor. For
example, every answer you gave with its reason:

```sql
select e.prompt, f.true_label, f.reason, e.images
from feedback f join evaluations e on e.id = f.evaluation_id
where f.reason <> '';
```

---

# Hosting it online (Railway)

The hosted app needs a login for every page and API call, using Supabase Auth. Each answer, task and run
records who made it (`labelled_by`, `created_by`, `started_by`).

## 1. Supabase: only invited people can sign in

1. **Authentication > Sign In / Providers > Email**: keep email on, **turn off "Allow new users to sign up"**.
2. **Authentication > Users > Add user** (or Invite): create an account for yourself and for each labeller.
3. **Settings > API Keys**: copy the **anon / publishable** key. This one is public by design (it can only
   sign people in); it is different from the service key.

## 2. Railway

1. Put this folder in a **private** GitHub repo (`.env` and `data/` are already git-ignored; never commit them).
2. Railway > New Project > Deploy from GitHub repo. It builds from the `Dockerfile` and checks `/healthz`.
3. In the service's **Variables**, add (values from your accounts; paste them into Railway, not into chat):

| Variable | Value |
|---|---|
| `IMAGE_JUDGE_AUTH` | `1` |
| `IMAGE_JUDGE_DATABASE_URL` | Supabase session-pooler connection string |
| `SUPABASE_URL` | `https://<ref>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | service_role key (server only; only needed if images are on Supabase Storage) |
| `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_REGION` | Cloudflare R2 / Backblaze B2 image bucket (recommended, see 2b) |
| `SUPABASE_ANON_KEY` | anon / publishable key |
| `ANTHROPIC_API_KEY` | your mwapi key |
| `ANTHROPIC_BASE_URL` | `https://api.mwapi.dev` |
| `IMAGE_JUDGE_MODEL` | `claude-sonnet-5` |
| `IMAGE_JUDGE_RUBRIC` | `v3` |
| `IMAGE_JUDGE_RUNS` | `2` |
| `IMAGE_JUDGE_FALLBACKS` | `0` |
| `IMAGE_JUDGE_MAX_CONCURRENCY` | `2` |
| `IMAGE_JUDGE_ALLOWED_EMAILS` | optional, comma-separated; empty = everyone you invited |

4. Service > Settings > Networking > **Generate Domain**, then open it and sign in.

## Things to know

- **Run one instance only.** Benchmark and training runs are background jobs held in memory. A redeploy or
  restart ends a running job (it is marked "interrupted"); start it again.
- **The `claude-code:` and `codex:` models do not exist online.** They need the CLI on your own PC. Hosted,
  the judge uses the API model above.
- **Importing a dataset from a folder path is disabled** when hosted (it would read the server's disk). Upload
  tasks through the browser; use `scripts/migrate_to_supabase.py` once, from your PC, for existing data.
- Anyone you invite can read and change all data. Per-person data separation is not built; the
  `labelled_by` columns let you filter by person when training.
