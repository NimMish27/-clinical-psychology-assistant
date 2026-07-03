const BASE = "";

export interface ChatResponse {
  answer: string;
  sources: Array<{
    text: string;
    source: string;
    page: number;
    score: number;
  }>;
  model: string;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface IngestResponse {
  status: "success" | "partial" | "failed";
  total_files: number;
  succeeded: number;
  failed: number;
  total_chunks: number;
  files: Array<{
    filename: string;
    status: string;
    pages_extracted: number;
    chunks_created: number;
    chunks_embedded: number;
    chunks_stored: number;
    error: string | null;
  }>;
  elapsed_ms: number;
  collection: string;
}

export async function sendChatMessage(
  query: string,
  n_sources = 5,
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, n_sources }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail?.[0]?.msg || err?.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function uploadFiles(files: File[]): Promise<IngestResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${BASE}/api/v1/ingest`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
