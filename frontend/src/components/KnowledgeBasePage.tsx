import { useState, useRef } from "react";
import { uploadFiles } from "../api/client";

interface StatusItem {
  filename: string;
  status: "pending" | "success" | "error";
  detail: string;
}

export default function KnowledgeBasePage() {
  const [dragOver, setDragOver] = useState(false);
  const [statuses, setStatuses] = useState<StatusItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

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
      const res = await uploadFiles(fileArray);
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
      <p style={{ fontSize: "0.875rem", color: "var(--color-text-secondary)", marginBottom: 16 }}>
        Upload PDF documents to expand the knowledge base. Files are processed, chunked, embedded,
        and stored in the vector database for retrieval during chat.
      </p>

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
