# Gemini Sandbox

A terminal interface for the Google Gemini API (paid AI Studio tier). Built for developers who want a single CLI that handles chat, file uploads, image generation, code execution, embeddings, URL fetching, YouTube transcript analysis, and a two-agent deliberation engine — all with explicit local spend tracking so a runaway request doesn't burn your balance silently.

This is a personal sandbox project published as-is. It works on Windows, macOS, and Linux. The only dependency you need to set up yourself is a Google AI Studio API key.

## Quick start

```bash
git clone https://github.com/RaginGonzo/gemini-sandbox.git
cd gemini-sandbox
pip install -r requirements.txt
cp .env.example .env
# Edit .env and paste your Gemini API key
python gemini_sandbox.py
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>. Paid usage is billed per token; see <https://ai.google.dev/pricing>.

## Setup

1. **Python 3.9 or newer.**
2. **API keys.** Copy `.env.example` to `.env` and fill in:
   - `GEMINI_API_KEY` — required. From AI Studio.
   - `YOUTUBE_API_KEY` — optional. Only needed if you want to use `/yt_search`. Get one in Google Cloud Console under YouTube Data API v3.
3. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run it.**
   ```bash
   python gemini_sandbox.py
   ```

On first launch the script asks for a starting balance in USD. This is a local counter the script uses to track spend against — it doesn't query Google's billing API. Match it to whatever you've prepaid on AI Studio and use `/sync` to reconcile when needed.

## Personalize the model's context

The script feeds a small `USER_MEMORY` block to the model on every turn so it knows who it's working with — your role, hardware, current project, and focus. The default block has bracketed placeholders. Open `gemini_sandbox.py`, find the `USER_MEMORY` constant near the top of the file (around line 200), and replace the placeholders with your own info. Anything in this block is included silently in every system prompt — keep it short and factual.

If you'd rather not personalize, leave the placeholders alone — the script still runs, the model just won't tailor its tone or examples to you.

## Commands

| Command | What it does |
| --- | --- |
| `/quit` | Save logs and exit. |
| `/clear` | Clear the terminal. |
| `/reset` | Restore default system prompt, keep history. |
| `/reset hard` | Restore default system prompt, wipe history. |
| `/sync` | Manually sync request/token counts from AI Studio dashboard. |
| `/model` | Swap models mid-session, history preserved. |
| `/system` <directive> | Replace the system prompt for this session. Wipes history. Custom prompt persists across /model swaps. |
| `/upload <path>` | Inject a file (image, doc, code) into context. |
| `/upload run <file.py>` | Execute a Python file in the sandbox and inject the output silently into chat memory. |
| `/imagine <prompt>` | Generate an image with Gemini 2.5 Flash Image. |
| `/imagine pro <prompt>` | Generate an image with Gemini 3 Pro Image (higher quality). |
| `/url <url>` | Fetch and integrate a URL into context. |
| `/execute <prompt>` | Run sandboxed Python iteratively (with web grounding). |
| `/embed <text>` | Generate an embedding vector; shows cosine similarity to previous embed. |
| `/history` | Show context size and session totals. |
| `/balance` | Open the balance menu. |
| `/add <amount>` | Top up local balance counter. |
| `/thinkon` / `/thinkoff` | Toggle extended thinking. |
| `/showthink` / `/hidethink` | Toggle visibility of the reasoning stream. |
| `/yt_search <query>` | YouTube search via Data API v3. |
| `/yt_analyze <video_id>` | Pull a transcript and feed it to the model for analysis. |
| `/council` | Run a two-agent deliberation (Noel chairs, Eli challenges). |

## What it does well

- **Spend visibility.** Every turn shows tokens in/out, message cost, session total, and remaining balance. Yellow/red warnings before you bottom out.
- **Atomic balance tracking.** File-locked credit ledger so a crashed turn doesn't desync your books.
- **Partial-token billing.** If a stream fails mid-response, you're billed for what the model actually produced.
- **Search grounding tracker.** Tracks free-tier grounding quota separately for 2.x (daily) and 3.x (monthly) families.
- **History pruning.** Auto-trims at 450 turns to keep context from exploding.
- **Prompt-injection guard on uploads.** Operator-level guard inserted before the file payload so a malicious document can't override the system prompt on the upload turn.
- **Streaming everywhere.** Text, embeddings, image, code execution — all stream where the API supports it.

## Models supported

Chat: `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite`, `gemini-3.1-pro-preview`

Image: `gemini-2.5-flash-image` (default), `gemini-3-pro-image-preview` (`/imagine pro`)

Embedding: `gemini-embedding-001`

Preview-tier models require preview access on your project. If a model returns 404, swap to a stable one with `/model`.

## Files the script creates

The script writes to the working directory:

- `Chat_Logs/` — markdown transcript per session
- `Generated_Images/` — saved images from `/imagine`
- `Code_Results/` — `/execute` and `/upload run` outputs
- `Council_Logs/` — `/council` transcripts
- `gemini_credits.json` — local balance counter
- `gemini_telemetry.json` — request/token counters

All of these are listed in `.gitignore` so secrets and runtime state don't get committed.

## Security notes

- **Never commit `.env`.** The included `.gitignore` excludes it, but double-check before your first push.
- **Restrict the API key.** In AI Studio or Cloud Console, restrict the key to the Generative Language API. Google requires this from June 19, 2026.
- **Don't paste the key in client-side code.** This is a backend/CLI script — keep it that way.
- **Rotate keys.** A 90-day rotation cadence is reasonable.

## Known limitations

- The credit counter is local-only. It does not query Google's billing API. Use `/sync` to reconcile against the AI Studio dashboard.
- Image-generation tokens are not counted in the session token total — only the flat per-image cost is added.
- Preview models can have stricter rate limits and may return 404 if your project doesn't have access. Swap to a stable model with `/model`.
- YouTube transcript fetching depends on `youtube-transcript-api`; not every video has transcripts available.

## License

GPLv3. See `LICENSE`.

## Acknowledgments

Built on top of the [google-genai](https://github.com/googleapis/python-genai) SDK and [rich](https://github.com/Textualize/rich) for the terminal UI.
