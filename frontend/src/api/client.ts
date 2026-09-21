const BASE = "";

export type Framework = "cbt" | "act" | "humanistic";

export const FRAMEWORKS: Array<{ id: Framework; label: string }> = [
  { id: "cbt", label: "CBT" },
  { id: "act", label: "ACT" },
  { id: "humanistic", label: "Humanistic" },
];

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

export interface UploadOptions {
  collection?: string;
  user?: string;
  project?: string;
  documentType?: string;
}

export interface KbCollectionInfo {
  framework: string;
  built: boolean;
  collection: string | null;
  index_version: string | null;
  chunk_count: number | null;
  built_at: string | null;
  books: string[];
}

export interface KbRebuildResponse {
  status: "ok" | "failed";
  built_at: string | null;
  source_files: string[];
  parents: number;
  children: number;
  total_tokens: number;
  errors: string[];
  skipped: string[];
  collections: Array<{
    framework: string;
    collection: string;
    chunk_count: number;
  }>;
}

export interface KbDocumentSummary {
  source: string;
  chunks: number;
  user: string | null;
  project: string | null;
  document_type: string | null;
  upload_date: string | null;
}

export interface KbDocumentsResponse {
  collection: string;
  sources: KbDocumentSummary[];
}

export interface EvidenceItem {
  source: string;
  content: string;
  quality_rating: number | null;
  framework: string | null;
  chapter: string | null;
  topic: string | null;
  intervention: string | null;
  symptom: string | null;
  population: string | null;
  page: number;
  chunk_id: string;
  is_user_document: boolean;
}

export interface EvidenceSearchResponse {
  query: string;
  framework: string | null;
  sources: EvidenceItem[];
  synthesis: string | null;
  key_findings: string[];
  evidence_gaps: string[];
  confidence: number | null;
  total_candidates: number;
  elapsed_ms: number;
  user_documents_included: boolean;
  user_documents_returned: number;
}

export interface EvidenceSearchOptions {
  framework?: Framework | null;
  includeUserDocuments?: boolean;
  nResults?: number;
  chapter?: string;
  topic?: string;
  intervention?: string;
  symptom?: string;
  population?: string;
}

async function parseError(res: Response): Promise<string> {
  const err = await res.json().catch(() => null);
  if (!err) return `HTTP ${res.status}`;
  if (typeof err.message === "string") return err.message;
  if (typeof err.detail === "string") return err.detail;
  if (Array.isArray(err.detail) && err.detail[0]?.msg) return err.detail[0].msg;
  return `HTTP ${res.status}`;
}

export async function sendChatMessage(
  query: string,
  n_sources = 5,
  framework?: Framework | null,
  includeUserDocuments = false,
): Promise<ChatResponse> {
  const body: Record<string, unknown> = { query, n_sources };
  if (framework) body.framework = framework;
  if (includeUserDocuments) body.include_user_documents = true;

  const res = await fetch(`${BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function uploadFiles(
  files: File[],
  options: UploadOptions = {},
): Promise<IngestResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  if (options.collection) form.append("collection", options.collection);
  if (options.user) form.append("user", options.user);
  if (options.project) form.append("project", options.project);
  if (options.documentType) form.append("document_type", options.documentType);

  const res = await fetch(`${BASE}/api/v1/ingest`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchKbCollections(): Promise<{
  collections: KbCollectionInfo[];
}> {
  const res = await fetch(`${BASE}/api/v1/kb/collections`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function rebuildKnowledgeBase(
  framework?: Framework | null,
): Promise<KbRebuildResponse> {
  const res = await fetch(`${BASE}/api/v1/kb/rebuild`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(framework ? { framework } : {}),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchKbDocuments(
  collection = "user_documents",
): Promise<KbDocumentsResponse> {
  const res = await fetch(
    `${BASE}/api/v1/kb/documents?collection=${encodeURIComponent(collection)}`,
  );
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function deleteKbDocument(
  source: string,
  collection = "user_documents",
): Promise<{ collection: string; source: string; deleted: number }> {
  const res = await fetch(
    `${BASE}/api/v1/kb/documents?collection=${encodeURIComponent(
      collection,
    )}&source=${encodeURIComponent(source)}`,
    { method: "DELETE" },
  );
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function searchEvidence(
  query: string,
  options: EvidenceSearchOptions = {},
): Promise<EvidenceSearchResponse> {
  const body: Record<string, unknown> = { query, n_results: options.nResults ?? 5 };
  if (options.framework) body.framework = options.framework;
  if (options.includeUserDocuments) body.include_user_documents = true;
  for (const key of ["chapter", "topic", "intervention", "symptom", "population"] as const) {
    const value = options[key];
    if (value) body[key] = value;
  }

  const res = await fetch(`${BASE}/api/v1/evidence/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}
