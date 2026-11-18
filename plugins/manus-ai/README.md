# manus-ai (ZCode plugin)

MCP server plugin that lets ZCode drive [Manus AI](https://manus.im) — the autonomous agent
service — through its official API (`https://api.manus.ai`, docs at
[open.manus.ai/docs](https://open.manus.ai/docs)).

Launch Manus agent tasks (web research, coding jobs, document production), answer the agent's
questions mid-run, stop it, and collect results — all without leaving ZCode.

## Tools

| Tool | What it does |
|---|---|
| `manus_me` | Verify the API credential; returns your Manus user ID |
| `manus_create_task` | Start an agent task from a prompt (options: title, `lite`/`standard`/`max` profile, attachments by URL, project, locale, structured output schema) |
| `manus_get_task` | Get a task's status and metadata |
| `manus_list_tasks` | List your recent tasks (paginated) |
| `manus_list_messages` | Get a task's event history — agent outputs, final results, errors |
| `manus_send_message` | Follow up mid-run: answer a question or add instructions |
| `manus_stop_task` | Stop a running task |

Typical flow: `manus_create_task` → poll `manus_get_task` until status is `stopped` →
`manus_list_messages` to read the result. If status is `waiting`, the agent asked a question —
reply with `manus_send_message`.

## Setup: provide your API key

Create a key in the Manus webapp under **Manus Developers → API Keys** (shown only once).
Then make it available to the server in **one** of these ways:

1. **File (easiest on Windows):** paste the key as the first line of
   `C:\Users\<you>\.manus\api_key` (create the `.manus` folder if needed).
2. **Environment variable:** `setx MANUS_API_KEY "your-key"` (system-wide), then restart ZCode.

Never commit the key to this repository.

## Requirements

- Node.js ≥ 18 on PATH (`node --version` to check). The server has zero npm dependencies.

## Troubleshooting

- **`unauthenticated` / 401** — the key is wrong, revoked, or the env var was set after ZCode
  started (restart ZCode). Run the `manus_me` tool to re-check.
- **`rate_limited`** — rate limits are per user across all your keys; wait and retry.
- **Free personal accounts** — task creation is forced to the `lite` profile regardless of
  what you pass.
- **`invalid_argument` on `structured_output_schema`** — the schema must be strict: every
  property in `required`, `additionalProperties: false`.
