import { useRef, useState, useEffect } from "react";
import {
  deleteKbDocument,
  fetchKbCollections,
  fetchKbDocuments,
  rebuildKnowledgeBase,
  uploadFiles,
  type KbCollectionInfo,
  type KbDocumentSummary,
} from "../api/client";
import { Badge, Button, Meter, Panel, Skeleton, useToast } from "../ui/primitives";
import { AlertCircleIcon, CheckCircleIcon, ClockIcon, RefreshIcon, TrashIcon, UploadIcon } from "../ui/icons";

interface StatusItem {
  filename: string;
  status: "pending" | "success" | "error";
  detail: string;
}

const USER_COLLECTION = "user_documents";

export default function KnowledgeBasePage() {
  const toast = useToast();
  const [dragOver, setDragOver] = useState(false);
  const [statuses, setStatuses] = useState<StatusItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const [collections, setCollections] = useState<KbCollectionInfo[] | null>(null);
  const [rebuilding, setRebuilding] = useState(false);
  const [rebuildMsg, setRebuildMsg] = useState<string | null>(null);

  const [docs, setDocs] = useState<KbDocumentSummary[] | null>(null);
  const [docError, setDocError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const [user, setUser] = useState("");
  const [project, setProject] = useState("");
  const [documentType, setDocumentType] = useState("");

  async function refreshCollections() {
    try {
      const res = await fetchKbCollections();
      setCollections(res.collections);
    } catch {
      setCollections([]);
    }
  }

  async function refreshDocs() {
    try {
      const res = await fetchKbDocuments(USER_COLLECTION);
      setDocs(res.sources);
      setDocError(null);
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Failed to load documents");
    }
  }

  useEffect(() => {
    refreshCollections();
    refreshDocs();
  }, []);

  async function handleRebuild() {
    setRebuilding(true);
    setRebuildMsg(null);
    try {
      const res = await rebuildKnowledgeBase();
      if (res.status === "ok") {
        const msg = `Rebuilt ${res.collections.length} knowledge base${res.collections.length === 1 ? "" : "s"} — ${res.children} chunks.`;
        setRebuildMsg(msg);
        toast.push("success", "Knowledge base rebuilt", msg);
      } else {
        const msg = `Build failed: ${res.errors.join("; ")}`;
        setRebuildMsg(msg);
        toast.push("error", "Build failed", msg);
      }
      await refreshCollections();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setRebuildMsg(`Rebuild failed: ${msg}`);
      toast.push("error", "Rebuild failed", msg);
    } finally {
      setRebuilding(false);
    }
  }

  async function handleDelete(source: string) {
    setDeleting(source);
    try {
      await deleteKbDocument(source, USER_COLLECTION);
      await refreshDocs();
      toast.push("success", "Document deleted", source);
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  }

  async function handleFiles(files: FileList | File[]) {
    const fileArray = Array.from(files).filter(
      (f) => f.type === "application/pdf" || f.name.endsWith(".pdf"),
    );
    if (fileArray.length === 0) {
      toast.push("error", "No PDF files selected", "Only PDF files are accepted");
      return;
    }

    const initial: StatusItem[] = fileArray.map((f) => ({
      filename: f.name,
      status: "pending" as const,
      detail: "Uploading…",
    }));
    setStatuses((s) => [...s, ...initial]);
    setUploading(true);

    try {
      const res = await uploadFiles(fileArray, {
        collection: USER_COLLECTION,
        user: user.trim() || undefined,
        project: project.trim() || undefined,
        documentType: documentType.trim() || undefined,
      });
      const updated = fileArray.map((f) => {
        const fileResult = res.files.find((fr) => fr.filename === f.name);
        if (fileResult?.status === "success") {
          return {
            filename: f.name,
            status: "success" as const,
            detail: `${fileResult.chunks_stored || 0} chunks from ${fileResult.pages_extracted || 0} pages`,
          };
        }
        return {
          filename: f.name,
          status: "error" as const,
          detail: fileResult?.error || `Upload ${res.status}`,
        };
      });

      setStatuses((s) => {
        const copy = [...s];
        for (let i = 0; i < updated.length; i++) {
          const idx = copy.length - updated.length + i;
          if (idx >= 0) copy[idx] = updated[i];
        }
        return copy;
      });
      await refreshDocs();
    } catch (err) {
      setStatuses((s) => {
        const copy = [...s];
        for (let i = 0; i < fileArray.length; i++) {
          const idx = copy.length - fileArray.length + i;
          if (idx >= 0)
            copy[idx] = {
              filename: fileArray[i].name,
              status: "error" as const,
              detail: err instanceof Error ? err.message : "Upload failed",
            };
        }
        return copy;
      });
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) handleFiles(e.target.files);
    e.target.value = "";
  }

  return (
    <div className="canvas canvas-narrow">
      <div className="section-head">
        <div>
          <div className="kicker">Knowledge</div>
          <h1 className="section-title">Knowledge Base</h1>
          <p className="section-sub">
            Curated framework literature, rebuilt from the repo, plus your private document collection.
          </p>
        </div>
      </div>

      {/* Framework Knowledge */}
      <Panel
        title={
          <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--space-2)" }}>
            Framework Knowledge <Badge tone="neutral">read-only</Badge>
          </span>
        }
      >
        {collections === null ? (
          <div style={{ padding: "var(--space-4) var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            <Skeleton width="100%" height={40} />
            <Skeleton width="100%" height={40} />
          </div>
        ) : (
          collections.map((c) => (
            <div key={c.framework} className="kb-build-card">
              <div className="kb-build-meta">
                <div className="kb-build-name">
                  <span style={{ textTransform: "uppercase", fontWeight: 650 }}>{c.framework}</span>
                  {c.built ? (
                    <Badge tone="success" dot>
                      Built
                    </Badge>
                  ) : (
                    <Badge tone="neutral" dot>
                      Not built
                    </Badge>
                  )}
                </div>
                <div className="kb-build-sub">
                  {c.built ? (
                    <>
                      {c.collection} · {c.chunk_count ?? 0} chunks · v{c.index_version} · {c.built_at}
                    </>
                  ) : (
                    "No curated sources indexed yet — run a rebuild."
                  )}
                </div>
              </div>
              {c.built && (
                <div style={{ width: 120 }}>
                  <Meter value={100} showValue={false} />
                </div>
              )}
            </div>
          ))
        )}
        <div className="list-row" style={{ background: "var(--c-bg-subtle)" }}>
          <div className="row-main">
            <div className="row-title" style={{ fontSize: "var(--text-sm)" }}>
              Rebuild curated knowledge bases from the repository
            </div>
            <div className="row-sub">Pipeline: validate → build → chunk → embed → index</div>
          </div>
          <Button variant="primary" size="sm" disabled={rebuilding} onClick={handleRebuild}>
            <RefreshIcon /> {rebuilding ? "Rebuilding…" : "Rebuild"}
          </Button>
        </div>
        {rebuildMsg && (
          <div className="list-row" style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-2)" }}>
            {rebuildMsg}
          </div>
        )}
      </Panel>

      {/* My Documents */}
      <Panel
        title={
          <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--space-2)" }}>
            My Documents <Badge tone="warning">private</Badge>
          </span>
        }
      >
        <div className="field" style={{ padding: "var(--space-4) var(--space-5)" }}>
          <div className="field-label">Upload metadata (optional)</div>
          <div className="upload-meta-fields" style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
            <input className="input" type="text" placeholder="User" value={user} onChange={(e) => setUser(e.target.value)} style={{ flex: 1, minWidth: 140 }} />
            <input className="input" type="text" placeholder="Project" value={project} onChange={(e) => setProject(e.target.value)} style={{ flex: 1, minWidth: 140 }} />
            <input className="input" type="text" placeholder="Document type" value={documentType} onChange={(e) => setDocumentType(e.target.value)} style={{ flex: 1, minWidth: 140 }} />
          </div>
          <p className="field-hint" style={{ marginTop: 4 }}>
            Tagged with user / project / document_type / upload_date. Only included in results when “Include my documents” is enabled.
          </p>
        </div>

        <div style={{ padding: "var(--space-4) var(--space-5)" }}>
          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => !uploading && inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <UploadIcon />
            <div className="upload-title">{uploading ? "Uploading…" : "Drop PDFs here or click to browse"}</div>
            <p>Only PDF files are accepted</p>
            <input
              ref={inputRef}
              className="file-input"
              type="file"
              multiple
              accept=".pdf,application/pdf"
              onChange={handleInputChange}
            />
          </div>
        </div>

        {statuses.length > 0 && (
          <div className="checklist">
            {statuses.map((s, i) => (
              <div key={i} className="checklist-item">
                <span className="check done">
                  {s.status === "success" ? <CheckCircleIcon /> : s.status === "error" ? <AlertCircleIcon /> : <ClockIcon />}
                </span>
                <div>
                  <div className="checklist-title" style={{ fontSize: "var(--text-sm)" }}>{s.filename}</div>
                  <div className="checklist-desc">{s.detail}</div>
                </div>
                <span style={{ marginLeft: "auto" }}>
                  <Badge tone={s.status === "success" ? "success" : s.status === "error" ? "danger" : "warning"} dot>
                    {s.status}
                  </Badge>
                </span>
              </div>
            ))}
          </div>
        )}

        {docError && <ErrorNote>{docError}</ErrorNote>}

        {docs !== null && (
          <Panel title={`Uploaded documents (${docs.length})`}>
            {docs.length === 0 ? (
              <div className="empty-state" style={{ padding: "var(--space-8)" }}>
                <UploadIcon />
                <h3>No documents yet</h3>
                <p>Upload PDFs above to start building your private collection.</p>
              </div>
            ) : (
              <div className="checklist">
                {docs.map((d) => (
                  <div key={d.source} className="checklist-item">
                    <div>
                      <div className="checklist-title" style={{ fontSize: "var(--text-sm)" }}>{d.source}</div>
                      <div className="checklist-desc">
                        {d.chunks} chunk{d.chunks === 1 ? "" : "s"}
                        {d.user ? ` · ${d.user}` : ""}
                        {d.project ? ` · ${d.project}` : ""}
                        {d.document_type ? ` · ${d.document_type}` : ""}
                        {d.upload_date ? ` · ${d.upload_date}` : ""}
                      </div>
                    </div>
                    <span style={{ marginLeft: "auto" }}>
                      <Button variant="danger" size="sm" disabled={deleting === d.source} onClick={() => handleDelete(d.source)}>
                        <TrashIcon /> {deleting === d.source ? "Deleting…" : "Delete"}
                      </Button>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        )}
      </Panel>
    </div>
  );
}

function ErrorNote({ children }: { children: React.ReactNode }) {
  return (
    <div className="error-state" style={{ margin: "var(--space-4) var(--space-5)" }}>
      <AlertCircleIcon />
      <div>
        <div className="error-title">Something went wrong</div>
        <p>{children}</p>
      </div>
    </div>
  );
}
