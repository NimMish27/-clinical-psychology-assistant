import { useState, useRef, useEffect } from "react";
import {
  deleteKbDocument,
  fetchKbCollections,
  fetchKbDocuments,
  rebuildKnowledgeBase,
  uploadFiles,
  type KbCollectionInfo,
  type KbDocumentSummary,
} from "../api/client";

interface StatusItem {
  filename: string;
  status: "pending" | "success" | "error";
  detail: string;
}

const USER_COLLECTION = "user_documents";

export default function KnowledgeBasePage() {
  const [dragOver, setDragOver] = useState(false);
  const [statuses, setStatuses] = useState<StatusItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const [collections, setCollections] = useState<KbCollectionInfo[]>([]);
  const [rebuilding, setRebuilding] = useState(false);
  const [rebuildMsg, setRebuildMsg] = useState<string | null>(null);

  const [docs, setDocs] = useState<KbDocumentSummary[]>([]);
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
        setRebuildMsg(
          `Rebuilt ${res.collections.length} knowledge base${res.collections.length === 1 ? "" : "s"} — ${res.children} chunks.`,
        );
      } else {
        setRebuildMsg(`Build failed: ${res.errors.join("; ")}`);
      }
      await refreshCollections();
    } catch (err) {
      setRebuildMsg(`Rebuild failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setRebuilding(false);
    }
  }

  async function handleDelete(source: string) {
    setDeleting(source);
    try {
      await deleteKbDocument(source, USER_COLLECTION);
      await refreshDocs();
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
      setStatuses((s) => [
        ...s,
        { filename: "No PDF files selected", status: "error", detail: "Only PDF files are accepted" },
      ]);
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

  function handleZoneClick() {
    if (!uploading) inputRef.current?.click();
  }

  return (
    <div className="kb-container">
      <h3>Knowledge Base</h3>

      <section className="kb-section">
        <h4>Framework Knowledge</h4>
        <p className="kb-hint">
          Curated literature committed to the repository and rebuilt through the
          build pipeline. Read-only in the UI.
        </p>

        <div className="kb-framework-list">
          {collections.map((c) => (
            <div key={c.framework} className="kb-framework-card">
              <div className="kb-framework-name">
                {c.framework.toUpperCase()}
                {c.built && (
                  <span className="status-badge success">Built</span>
                )}
              </div>
              <div className="kb-framework-meta">
                {c.built ? (
                  <>
                    <span>collection: {c.collection}</span>
                    <span>{c.chunk_count ?? 0} chunks</span>
                    <span>version: {c.index_version}</span>
                  </>
                ) : (
                  <span>Not built — no curated sources indexed yet</span>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="kb-rebuild-row">
          <button
            className="send-btn"
            onClick={handleRebuild}
            disabled={rebuilding}
          >
            {rebuilding ? "Rebuilding…" : "Rebuild Knowledge Base"}
          </button>
          {rebuildMsg && <span className="kb-rebuild-msg">{rebuildMsg}</span>}
        </div>
      </section>

      <section className="kb-section">
        <h4>My Documents</h4>
        <p className="kb-hint">
          Upload PDFs to your private document collection. They are stored
          separately and only included in answers when you enable
          “Include my documents” in chat.
        </p>

        <div className="upload-meta-fields">
          <input
            className="chat-input"
            type="text"
            placeholder="User (optional)"
            value={user}
            onChange={(e) => setUser(e.target.value)}
          />
          <input
            className="chat-input"
            type="text"
            placeholder="Project (optional)"
            value={project}
            onChange={(e) => setProject(e.target.value)}
          />
          <input
            className="chat-input"
            type="text"
            placeholder="Document type (optional)"
            value={documentType}
            onChange={(e) => setDocumentType(e.target.value)}
          />
        </div>

        <div
          className={`upload-zone ${dragOver ? "drag-over" : ""}`}
          onClick={handleZoneClick}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <UploadIcon />
          <p className="upload-title">
            {uploading ? "Uploading…" : "Drop PDFs here or click to browse"}
          </p>
          <p>Only PDF files are accepted (max 50 MB each)</p>
          <input
            ref={inputRef}
            className="file-input"
            type="file"
            multiple
            accept=".pdf,application/pdf"
            onChange={handleInputChange}
          />
        </div>

        {statuses.length > 0 && (
          <div className="upload-status">
            {statuses.map((s, i) => (
              <div key={i} className="status-card">
                <span className={`status-icon ${s.status}`}>
                  {s.status === "success" ? (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12" /></svg>
                  ) : s.status === "error" ? (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>
                  ) : (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
                  )}
                </span>
                <div className="status-info">
                  <div className="filename">{s.filename}</div>
                  <div className="detail">{s.detail}</div>
                </div>
                <span className={`status-badge ${s.status}`}>
                  {s.status === "success" ? "Success" : s.status === "error" ? "Failed" : "Pending"}
                </span>
              </div>
            ))}
          </div>
        )}

        {docError && <div className="kb-rebuild-msg">{docError}</div>}

        {docs.length > 0 && (
          <div className="kb-doc-list">
            <h5>Uploaded documents</h5>
            {docs.map((d) => (
              <div key={d.source} className="kb-doc-row">
                <div className="kb-doc-info">
                  <div className="filename">{d.source}</div>
                  <div className="detail">
                    {d.chunks} chunk{d.chunks === 1 ? "" : "s"}
                    {d.user ? ` · user: ${d.user}` : ""}
                    {d.project ? ` · project: ${d.project}` : ""}
                    {d.document_type ? ` · type: ${d.document_type}` : ""}
                  </div>
                </div>
                <button
                  className="kb-delete-btn"
                  onClick={() => handleDelete(d.source)}
                  disabled={deleting === d.source}
                >
                  {deleting === d.source ? "Deleting…" : "Delete"}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}
