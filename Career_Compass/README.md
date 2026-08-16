# Career Compass

A local job-tracking portal that checks selected organization career feeds, finds target roles posted within seven days, records submitted interests, and tailors resumes for selected roles.

## Start it locally (recommended)

1. Copy `.env.example` to `.env`, then set `OPENAI_API_KEY` to enable tailoring.
2. Install the project dependencies once:

   `python -m pip install -r requirements.txt`

3. Start the portal:

   `python run.py`

4. Open `http://localhost:8000`.

Docker remains available as an optional, self-contained run method: `docker compose up --build`.

The first launch creates local `data/portal.json` for organizations, target roles, cached jobs, history, and resume metadata. Uploaded resumes and generated documents stay in `data/`, which Git ignores.

## Set up sources

Use **Settings** to add organizations and choose a source type:

- **Greenhouse**: copy the board token from `boards.greenhouse.io/<token>`.
- **Lever**: copy the site token from `jobs.lever.co/<token>`.
- **Generic page**: enter a public careers URL. The fallback reads public JSON-LD `JobPosting` records; bespoke ATS sites may need a dedicated connector.

See `config/organizations.example.json` for examples. Clicking refresh fetches all configured sources. It limits results to jobs with posting dates inside the last seven days and uses exact, keyword, and related role-family matching. Individual source errors are reported without blocking other sources.

## Resume tailoring

Upload a TXT, PDF, or DOCX base resume. The prompt is stored separately in `prompts/resume_rewrite_prompt.txt` and is editable in the UI. It must include `{role_name}` and `{organization_name}`. For each selected job, the backend calls the OpenAI Responses API with the base resume, job description, and filled prompt, then downloads a one-page DOCX named `Resume_Job_<Role>_<Organization>_<YYYY-MM-DD>.docx`.

The API key is read only from `.env` inside the container, never included in browser code or portal data. The integration follows the [official OpenAI API quickstart](https://platform.openai.com/docs/quickstart). Review every generated resume for accuracy, and respect career-site terms and rate limits.

### Google Careers browser connector
Google Careers may render job cards with JavaScript. After installing Python dependencies, install Chromium once:

```powershell
pip install -r requirements.txt
playwright install chromium
```

If `/api/refresh` reports that Chromium is missing, rerun the second command inside the activated virtual environment. Google jobs without a reliable published date are tracked by first-seen date; the first successful refresh shows matching current results, and later refreshes retain jobs first discovered within seven days.

## Location preferences
Each configured organization now has a required city preference. Career Compass validates the city when preferences are saved using the OpenStreetMap Nominatim search endpoint, stores the normalized city name, and filters refreshed jobs to postings whose displayed location matches that organization preference. Google Careers location is extracted from the browser-rendered result card. If a source does not expose a job location, that job is excluded when a location preference is active because the match cannot be verified.
