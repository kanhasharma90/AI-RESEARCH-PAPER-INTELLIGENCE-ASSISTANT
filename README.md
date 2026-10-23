# Paper Intelligence: frontend

React + Vite + TypeScript + Tailwind CSS 4 frontend for the AI Research Paper
Intelligence Assistant. It talks to the FastAPI backend in `api.py` over HTTP.
It never calls Gemini or the database directly.

## Run it (Windows PowerShell)

Terminal 1, backend (from the project folder; `GEMINI_API_KEY` must be set):

```powershell
cd "C:\Users\Kanha Sharma\Desktop\project"
uvicorn api:app --reload --port 8000
```

Terminal 2, frontend:

```powershell
cd "C:\Users\Kanha Sharma\Desktop\project\frontend"
npm install
npm run dev
```

Open http://localhost:5173. The dev server must stay on port 5173 because the
backend's CORS list only allows that origin (`strictPort` is on).

To point at a different backend, copy `.env.example` to `.env` and edit
`VITE_API_BASE_URL`. Only public values belong there: never database
credentials or the Gemini key.

## Pages

| Route | What it does | Backend endpoint |
|---|---|---|
| `/` Dashboard | Real counts, ask-a-question box | `GET /stats` |
| `/library` | Paginated paper list, filter by ID | `GET /papers` |
| `/chat` | Answers with citation, question and claim checks, expandable sources | `POST /chat` |
| `/search` | Ranked passages by semantic distance, no answer generation | `POST /search` |
| `/settings` | Shows the API URL, tests the connection | `GET /health` |

## Not built yet

- **PDF ingestion:** needs an upload endpoint and a single-job lock in the
  backend first.
- **Recent activity, failed-ingestion counts, "recently added":** the API exposes
  no timestamps, and `ingestion_failures.log` is overwritten each run.
- **Paper titles and detail view:** the `papers` table stores no titles.

## Structure

```
src/
  app/            router
  components/     layout/ ui/ papers/ chat/
  pages/          one file per route
  services/api.ts all HTTP calls and error handling
  types/api.ts    TypeScript shapes of the real API responses
  hooks/          useAsync, useDebouncedValue
```
