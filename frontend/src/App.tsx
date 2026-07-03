import { useState } from "react";
import Layout from "./components/Layout";
import ChatPage from "./components/ChatPage";
import KnowledgeBasePage from "./components/KnowledgeBasePage";

type Page = "chat" | "knowledge-base";

export default function App() {
  const [page, setPage] = useState<Page>("chat");

  return (
    <Layout page={page} onNavigate={setPage}>
      {page === "chat" ? <ChatPage /> : <KnowledgeBasePage />}
    </Layout>
  );
}
