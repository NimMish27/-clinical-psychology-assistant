import type { ReactNode } from "react";

type Page = "chat" | "knowledge-base";

interface LayoutProps {
  page: Page;
  onNavigate: (page: Page) => void;
  children: ReactNode;
}

export default function Layout({ page, onNavigate, children }: LayoutProps) {
  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Clinical Psychology Assistant</h1>
          <p>Evidence-based clinical support</p>
        </div>
        <nav className="sidebar-nav">
          <button
            className={`nav-item ${page === "chat" ? "active" : ""}`}
            onClick={() => onNavigate("chat")}
          >
            <ChatIcon />
            Chat
          </button>
          <button
            className={`nav-item ${page === "knowledge-base" ? "active" : ""}`}
            onClick={() => onNavigate("knowledge-base")}
          >
            <KBIcon />
            Knowledge Base
          </button>
        </nav>
        <div className="sidebar-footer">v0.1.0</div>
      </aside>
      <div className="main">
        <div className="main-content">{children}</div>
      </div>
    </>
  );
}

function ChatIcon() {
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function KBIcon() {
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}
