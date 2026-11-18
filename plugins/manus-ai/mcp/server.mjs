#!/usr/bin/env node
/**
 * Manus AI MCP server for ZCode.
 *
 * Zero-dependency Node (>= 18) stdio MCP server wrapping the official Manus API
 * (https://open.manus.ai/docs, base URL https://api.manus.ai).
 *
 * Credential resolution order:
 *   1. MANUS_API_KEY environment variable
 *   2. first line of ~/.manus/api_key
 *
 * Diagnostics go to stderr only; stdout carries the MCP JSON-RPC protocol.
 */
import { createInterface } from 'node:readline';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';

const BASE_URL = 'https://api.manus.ai';
const SERVER_INFO = { name: 'manus-ai', version: '0.1.0' };
const KNOWN_PROTOCOL_VERSIONS = ['2024-11-05', '2025-03-26', '2025-06-18'];
const REQUEST_TIMEOUT_MS = 90_000;

class ToolError extends Error {}

function apiKeyFilePath() {
  return join(homedir(), '.manus', 'api_key');
}

function resolveApiKey() {
  const fromEnv = (process.env.MANUS_API_KEY ?? '').trim();
  if (fromEnv) return fromEnv;
  try {
    const firstLine = readFileSync(apiKeyFilePath(), 'utf8').trim().split(/\r?\n/)[0].trim();
    if (firstLine) return firstLine;
  } catch {
    // file absent or unreadable — fall through
  }
  return null;
}

async function callManus(path, { method = 'GET', body, query } = {}) {
  const key = resolveApiKey();
  if (!key) {
    throw new ToolError(
      'No Manus API key found. Either set the MANUS_API_KEY environment variable system-wide ' +
        '(Windows: setx MANUS_API_KEY your-key, then restart ZCode) or paste the key as the ' +
        `first line of ${apiKeyFilePath()} (create the folder if needed). Keys are created in the ` +
        'Manus webapp under Manus Developers -> API Keys.'
    );
  }
  const url = new URL(BASE_URL + path);
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  }
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: {
        'x-manus-api-key': key,
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    throw new ToolError(`Could not reach ${BASE_URL}: ${err.cause?.code ?? err.name}: ${err.message}`);
  }
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = null;
  }
  if (!res.ok || data?.ok === false) {
    const code = data?.error?.code ?? `HTTP ${res.status}`;
    const message = data?.error?.message ?? text.slice(0, 500);
    throw new ToolError(`Manus API error (${code}): ${message}`);
  }
  return data;
}

function iso(seconds) {
  const n = Number(seconds);
  return Number.isFinite(n) ? new Date(n * 1000).toISOString() : undefined;
}

function stripEnvelope(data) {
  const { ok, request_id, ...rest } = data ?? {};
  return rest;
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

async function toolMe() {
  const data = stripEnvelope(await callManus('/v2/user.me'));
  return { ...data, note: 'API key is valid.' };
}

function buildCreateTaskBody(a) {
  const prompt = String(a.prompt ?? '').trim();
  if (!prompt) throw new ToolError('prompt is required');
  const body = { message: prompt };
  for (const url of Array.isArray(a.file_urls) ? a.file_urls : []) {
    if (typeof url !== 'string' || !url.trim()) throw new ToolError('file_urls entries must be non-empty URL strings');
    if (!Array.isArray(body.message)) body.message = [{ type: 'text', text: prompt }];
    body.message.push({ type: 'file', file_url: url });
  }
  const passthrough = [
    'title',
    'interactive_mode',
    'locale',
    'project_id',
    'share_visibility',
    'agent_profile',
    'structured_output_schema',
    'task_references',
  ];
  for (const field of passthrough) {
    if (a[field] !== undefined && a[field] !== null && a[field] !== '') body[field] = a[field];
  }
  return body;
}

async function toolCreateTask(a) {
  const data = stripEnvelope(await callManus('/v2/task.create', { method: 'POST', body: buildCreateTaskBody(a) }));
  return {
    ...data,
    hint: 'Poll manus_get_task for status; use manus_list_messages to collect output once status is stopped.',
  };
}

async function toolGetTask(a) {
  if (!a.task_id) throw new ToolError('task_id is required');
  const data = stripEnvelope(await callManus('/v2/task.detail', { query: { task_id: a.task_id } }));
  const task = data.task ?? {};
  return {
    ...task,
    created_at_iso: iso(task.created_at),
    updated_at_iso: iso(task.updated_at),
    status_meaning: {
      running: 'actively working',
      stopped: 'current run stopped — results ready unless has_running_background_jobs is true',
      waiting: 'paused, waiting for your input (answer via manus_send_message)',
      error: 'unrecoverable error — see manus_list_messages',
    }[task.status],
  };
}

async function toolListTasks(a) {
  const data = stripEnvelope(
    await callManus('/v2/task.list', {
      query: {
        limit: a.limit,
        cursor: a.cursor,
        order: a.order,
        scope: a.scope,
        project_id: a.project_id,
        agent_id: a.agent_id,
      },
    })
  );
  return {
    tasks: (data.data ?? []).map((t) => ({
      id: t.id,
      status: t.status,
      title: t.title,
      task_url: t.task_url,
      agent_profile: t.agent_profile,
      credit_usage: t.credit_usage,
      created_at: t.created_at,
      created_at_iso: iso(t.created_at),
      updated_at_iso: iso(t.updated_at),
    })),
    has_more: data.has_more,
    next_cursor: data.next_cursor,
  };
}

async function toolListMessages(a) {
  if (!a.task_id) throw new ToolError('task_id is required');
  const data = await callManus('/v2/task.listMessages', {
    query: {
      task_id: a.task_id,
      limit: a.limit,
      order: a.order,
      verbose: a.verbose === true ? 'true' : undefined,
    },
  });
  const events = Array.isArray(data) ? data : (data.data ?? data.events ?? []);
  return {
    events,
    hint:
      'status_update events carry agent_status (running | stopped | waiting | error); ' +
      'assistant_message events carry the agent\'s outputs including the final result; ' +
      'when agent_status is waiting, answer with manus_send_message.',
  };
}

async function toolSendMessage(a) {
  if (!a.task_id) throw new ToolError('task_id is required');
  const content = String(a.content ?? '').trim();
  if (!content) throw new ToolError('content is required');
  const data = stripEnvelope(
    await callManus('/v2/task.sendMessage', {
      method: 'POST',
      body: { task_id: a.task_id, message: { content } },
    })
  );
  return { ...data, hint: 'Message delivered. Poll manus_list_messages to see the agent resume.' };
}

async function toolStopTask(a) {
  if (!a.task_id) throw new ToolError('task_id is required');
  return stripEnvelope(await callManus('/v2/task.stop', { method: 'POST', body: { task_id: a.task_id } }));
}

// ---------------------------------------------------------------------------
// Tool declarations
// ---------------------------------------------------------------------------

const TOOLS = [
  {
    name: 'manus_me',
    description:
      'Verify the configured Manus AI API credential. Returns the Manus user ID the key belongs to. Use this first when Manus calls fail, to check authentication.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'manus_create_task',
    description:
      'Create a Manus AI agent task. An autonomous agent (browses the web, writes and runs code, produces documents) starts working on the prompt immediately. Returns the task ID, title and webapp URL. Consumes Manus credits.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'Instruction for the Manus agent. Roughly 5,000-token limit.' },
        title: { type: 'string', description: 'Optional custom task title.' },
        interactive_mode: {
          type: 'boolean',
          description: 'Allow the agent to pause and ask follow-up questions (then answer via manus_send_message). Default false.',
        },
        agent_profile: {
          type: 'string',
          enum: ['standard', 'lite', 'max'],
          description: 'Agent profile: standard (default), lite (faster/cheaper; forced for free accounts), max (deepest).',
        },
        locale: { type: 'string', description: 'Output language, e.g. "en", "zh-CN".' },
        project_id: { type: 'string', description: 'Associate the task with a Manus project; project instructions are auto-applied.' },
        share_visibility: {
          type: 'string',
          enum: ['private', 'team', 'public'],
          description: 'Who can view the share link. Default private; a share_url is only returned when not private.',
        },
        file_urls: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional public URLs of files to attach (each max 20 MB).',
        },
        task_references: {
          type: 'array',
          items: { type: 'string' },
          description: 'Up to 20 existing Manus task IDs to use as context.',
        },
        structured_output_schema: {
          type: 'object',
          description:
            'Optional JSON Schema for extracting a final structured result. Subset required: every object property listed in `required` and `additionalProperties: false`.',
        },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
  },
  {
    name: 'manus_get_task',
    description:
      'Get the current status and metadata of a Manus task. Statuses: running, stopped (done unless has_running_background_jobs is true), waiting (paused for your input), error. For actual results use manus_list_messages.',
    inputSchema: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID, e.g. "task_abc123".' },
      },
      required: ['task_id'],
      additionalProperties: false,
    },
  },
  {
    name: 'manus_list_tasks',
    description: 'List your Manus tasks, newest first by default, with pagination.',
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'integer', description: 'Tasks per page, 1-100. Default 20.' },
        cursor: { type: 'string', description: 'Pagination cursor from the previous call (next_cursor).' },
        order: { type: 'string', enum: ['asc', 'desc'], description: 'Sort by creation time. Default desc (newest first).' },
        scope: { type: 'string', enum: ['all', 'standard', 'project', 'agent_subtask'], description: 'Filter by task type. Default all.' },
        project_id: { type: 'string', description: 'Required when scope is "project".' },
        agent_id: { type: 'string', description: 'Required when scope is "agent_subtask".' },
      },
      additionalProperties: false,
    },
  },
  {
    name: 'manus_list_messages',
    description:
      "List a Manus task's event history: agent status changes, assistant messages (including the final result), and errors. Poll this while waiting for a task; once the latest status_update has agent_status=stopped, the last assistant_message holds the result.",
    inputSchema: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID, e.g. "task_abc123".' },
        limit: { type: 'integer', description: 'Maximum events to return.' },
        order: { type: 'string', enum: ['asc', 'desc'], description: 'Event order. Default desc.' },
        verbose: { type: 'boolean', description: 'Include tool_used events with parameters and results.' },
      },
      required: ['task_id'],
      additionalProperties: false,
    },
  },
  {
    name: 'manus_send_message',
    description:
      'Send a follow-up message to a Manus task: answer a question the agent asked (status "waiting"), confirm an action, or add new instructions. The agent resumes with this input.',
    inputSchema: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID, e.g. "task_abc123".' },
        content: { type: 'string', description: 'Message text for the agent.' },
      },
      required: ['task_id', 'content'],
      additionalProperties: false,
    },
  },
  {
    name: 'manus_stop_task',
    description: 'Stop a running Manus task. The agent halts its current run.',
    inputSchema: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID, e.g. "task_abc123".' },
      },
      required: ['task_id'],
      additionalProperties: false,
    },
  },
];

const TOOL_HANDLERS = {
  manus_me: toolMe,
  manus_create_task: toolCreateTask,
  manus_get_task: toolGetTask,
  manus_list_tasks: toolListTasks,
  manus_list_messages: toolListMessages,
  manus_send_message: toolSendMessage,
  manus_stop_task: toolStopTask,
};

// ---------------------------------------------------------------------------
// MCP JSON-RPC plumbing (stdio: one JSON message per line)
// ---------------------------------------------------------------------------

function send(message) {
  process.stdout.write(JSON.stringify(message) + '\n');
}

function textResult(text, isError = false) {
  const result = { content: [{ type: 'text', text }] };
  if (isError) result.isError = true;
  return result;
}

async function dispatchTool(name, args) {
  const handler = TOOL_HANDLERS[name];
  if (!handler) throw new ToolError(`Unknown tool: ${name}`);
  const payload = await handler(args ?? {});
  return textResult(JSON.stringify(payload, null, 2));
}

async function handleMessage(msg) {
  switch (msg.method) {
    case 'initialize': {
      const requested = msg.params?.protocolVersion;
      const protocolVersion = KNOWN_PROTOCOL_VERSIONS.includes(requested)
        ? requested
        : KNOWN_PROTOCOL_VERSIONS[KNOWN_PROTOCOL_VERSIONS.length - 1];
      return { protocolVersion, capabilities: { tools: {} }, serverInfo: SERVER_INFO };
    }
    case 'ping':
      return {};
    case 'tools/list':
      return { tools: TOOLS };
    case 'tools/call': {
      const name = msg.params?.name;
      try {
        return await dispatchTool(name, msg.params?.arguments);
      } catch (err) {
        return textResult(err instanceof Error ? err.message : String(err), true);
      }
    }
    default:
      return { error: { code: -32601, message: `Method not found: ${msg.method}` } };
  }
}

const rl = createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try {
    msg = JSON.parse(trimmed);
  } catch {
    return;
  }
  if (!msg || typeof msg !== 'object' || typeof msg.method !== 'string') return;
  if (msg.method.startsWith('notifications/')) return;
  handleMessage(msg)
    .then((result) => {
      if (result && result.error) {
        send({ jsonrpc: '2.0', id: msg.id, error: result.error });
      } else {
        send({ jsonrpc: '2.0', id: msg.id, result });
      }
    })
    .catch((err) => {
      send({
        jsonrpc: '2.0',
        id: msg.id,
        error: { code: -32603, message: err instanceof Error ? err.message : String(err) },
      });
    });
});
rl.on('close', () => process.exit(0));
