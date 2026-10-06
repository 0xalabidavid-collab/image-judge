# Putting Image Judge online for you and your labellers

Follow the parts in order. About an hour, mostly clicking in dashboards. **Never paste a key or password
into a chat or a document; only into the places named below.**

What you will end up with: one web address. You and the labellers sign in with email and password. Their
answers are saved to your Supabase database, images go to Cloudflare R2, and the judge runs on Railway.

| Part | Service | Cost (check current pricing) |
|---|---|---|
| Database + logins | Supabase | Free plan works; a project unused for a week is paused until you restore it |
| Image storage | Cloudflare R2 | Free up to 10 GB (your images are about 1 GB) |
| The app itself | Railway | A few dollars a month |
| The AI judge | your mwapi key | whatever your usage costs |

---

## Part 0 - Before you start

- [ ] **Rotate your mwapi key.** It was pasted into a chat. Make a new one with your provider and use only
      the new one below.
- [ ] Wait until the split experiment finishes (or stop it); do not restart the local app while it runs.
- [ ] Decide who is an **owner** (you) and list the labellers' emails.

---

## Part 1 - Supabase (database and logins)

1. Create the new Supabase account and a **new project**. Choose a region near you. **Write down the
   database password** you choose; it cannot be shown again.
2. **Settings > Database > Connection string**, choose **Session pooler**, copy the URI. Replace
   `[YOUR-PASSWORD]` in it with your database password. This is `IMAGE_JUDGE_DATABASE_URL`.
3. **Settings > API Keys**: copy the **Project URL** (`SUPABASE_URL`) and the **anon / publishable** key
   (`SUPABASE_ANON_KEY`). You will not need the service_role key if you use R2 for images.
4. **Authentication > Sign In / Providers > Email**: make sure Email is on and **turn OFF "Allow new users to
   sign up"**. Without this, anyone could create an account.
5. **Authentication > Users > Add user**: create an account (email + password, tick "auto confirm") for
   yourself and for each labeller. Send each labeller their password privately; they can keep it.

## Part 2 - Cloudflare R2 (images)

1. In Cloudflare: **R2 Object Storage > Create bucket**, named `image-judge`. Leave it private. (R2 asks for a
   payment method; nothing is charged below the free limits.)
2. **R2 > Manage API Tokens > Create API token**: permission **Object Read & Write**, limited to the
   `image-judge` bucket. Copy the **Access Key ID**, **Secret Access Key**, and your account's endpoint
   `https://<account-id>.r2.cloudflarestorage.com`.

## Part 3 - Copy your existing data

On your PC, add these lines to the file `image-judge\.env` (create the values from Parts 1 and 2):

```
IMAGE_JUDGE_DATABASE_URL=...
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
S3_ENDPOINT_URL=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET=image-judge
S3_REGION=auto
```

Then in PowerShell, from the `image-judge` folder:

```powershell
.\.venv\Scripts\python.exe scripts\migrate_to_supabase.py
```

That is a **dry run**: it only counts. Check it shows your tasks, answers and about 440 images. Then:

```powershell
.\.venv\Scripts\python.exe scripts\migrate_to_supabase.py --apply
```

It prints a table comparing local and Supabase counts, and finishes with "All copied and verified". If
anything is missing, run the same command again; it only copies what is missing. Your local data is not
changed.

> **From now on use only the online version.** If you keep using the local app too, your data splits in two.
> Treat the local folder as a backup.

(The one task whose image file is empty, "Place red background and pattern from input 2…", cannot be
copied. Re-upload it from the Judge page after launch.)

## Part 4 - Put the code on GitHub (private)

Railway builds from a GitHub repository. In PowerShell, from the `image-judge` folder:

```powershell
git init
git add .
git status
```

**Stop and read the `git status` list.** It must NOT contain `.env`, anything in `data\`, or `.venv\`
(they are already excluded). If you see them, do not continue; ask for help. Then:

```powershell
git commit -m "Image Judge"
```

On github.com create a **Private** repository (empty, no README), then run the two commands GitHub shows
under "push an existing repository" (`git remote add origin ...` and `git push -u origin main`).

## Part 5 - Railway (the app)

1. railway.com > **New Project > Deploy from GitHub repo** > pick the repository. It finds the `Dockerfile`
   and starts building.
2. Open the service > **Variables**. Add each of these (copy values from your `.env`; paste them here, nowhere
   else):

| Variable | Value |
|---|---|
| `IMAGE_JUDGE_AUTH` | `1` |
| `IMAGE_JUDGE_ADMIN_EMAILS` | **your** email (only owners can train and change lessons) |
| `IMAGE_JUDGE_DATABASE_URL` | from Part 1 |
| `SUPABASE_URL` | from Part 1 |
| `SUPABASE_ANON_KEY` | from Part 1 |
| `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_REGION` | from Part 2 |
| `ANTHROPIC_API_KEY` | your **new** mwapi key |
| `ANTHROPIC_BASE_URL` | `https://api.mwapi.dev` |
| `IMAGE_JUDGE_MODEL` | `claude-sonnet-5` |
| `IMAGE_JUDGE_RUBRIC` | `v3` |
| `IMAGE_JUDGE_RUNS` | `2` |
| `IMAGE_JUDGE_FALLBACKS` | `0` |
| `IMAGE_JUDGE_MAX_CONCURRENCY` | `2` |

3. **Settings > Networking > Generate Domain.** Wait for the deploy to show as healthy, then open the address.
4. Keep it at **one instance** (the default). Do not scale it up.

## Part 6 - Check it works, then invite people

Do these yourself first:

- [ ] The address shows a **Sign in** page, and you cannot see anything without signing in.
- [ ] Sign in with your email. The Judge page loads and the menu shows Train & Test and Benchmark.
- [ ] Open **Lessons**: your old versions and "4 of 67 answers have a reason" appear.
- [ ] Open a recent evaluation from the history: the images show (they come from R2).
- [ ] Run **one** new evaluation, mark it, and check it appears in the history.
- [ ] Sign in as a labeller (a private window): the menu has **no** Train & Test or Benchmark, and typing
      `/train` bounces back to the Judge page.
- [ ] Try signing up with a new email on the sign-in page: there is no sign-up, and Supabase refuses it.

Then send the labellers: the address, their email and password, and one line: **"Open Help first."**

---

## Running it day to day

- **Everything is saved online.** Nothing important lives on Railway; a redeploy loses nothing except a
  training run that was in progress (start it again).
- **Training runs:** only you can start them (Train & Test). They cost judge usage; the Lessons page shows
  the automatic tests the app runs by itself after every 5 corrections.
- **See who taught what:** in the Supabase **SQL editor**:

  ```sql
  select labelled_by, count(*) as answers, count(*) filter (where reason <> '') as with_reason
  from feedback group by labelled_by;
  ```
- **Add or remove a labeller:** Supabase > Authentication > Users.
- **Backups:** the free Supabase plan has no downloadable automatic backups, so every month export the main
  tables (`feedback`, `evaluations`, `set_tasks`, `knowledge`) as CSV from Supabase > Table editor, or upgrade
  the plan. The images are in R2. Your original local files from before the move are also still a backup.
- **Paused project:** if Supabase pauses it after a quiet week, open the Supabase dashboard and click
  **Restore**; the app works again within minutes.

## If something goes wrong

| What you see | Likely cause and fix |
|---|---|
| Railway deploy fails | Open the deploy log. A missing variable usually says which one. |
| Sign in page says "Wrong email or password" | The user does not exist in Supabase Authentication, or the password is wrong. |
| Login says the service is not available | `SUPABASE_URL` or `SUPABASE_ANON_KEY` is wrong in Railway. |
| Pages load but images are broken | R2 variables are wrong, or the bucket name differs. Check the Railway logs for "Cloudflare R2: ... failed". |
| "Anthropic API key missing or invalid" | Wrong key, or `ANTHROPIC_BASE_URL` not set to `https://api.mwapi.dev`. |
| Evaluate works locally but errors online | The online app can only use API models; `claude-code:` and `codex:` models need your own PC. |
| Everything is slow right after a quiet period | The first request wakes the server and the database; wait a minute. |
