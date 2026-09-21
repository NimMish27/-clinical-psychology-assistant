import { Badge, Panel, Switch } from "../ui/primitives";

export default function SettingsPage() {
  return (
    <div className="canvas canvas-narrow">
      <div className="section-head">
        <div>
          <div className="kicker">Settings</div>
          <h1 className="section-title">Settings</h1>
          <p className="section-sub">Application, knowledge base, and retrieval preferences.</p>
        </div>
      </div>

      <Panel title="General">
        <SettingRow
          title="Include my documents by default"
          desc="Pre-merge results from your private document collection into every evidence search."
          control={<Switch checked={false} onChange={() => {}} label="Include my documents by default" />}
        />
        <SettingRow
          title="Compact density"
          desc="Tighter spacing for high-volume case work. See the density preference test for details."
          control={<Switch checked={false} onChange={() => {}} label="Compact density" />}
        />
        <SettingRow
          title="Theme"
          desc="Warm stone light theme with indigo accent. Dark mode is on the roadmap."
          control={<Badge tone="neutral">Light</Badge>}
        />
      </Panel>

      <Panel title="Knowledge base">
        <SettingRow
          title="Framework literature"
          desc="Curated CBT / ACT / Humanistic knowledge bases are rebuilt from the repository."
          control={<Badge tone="success" dot>3 frameworks</Badge>}
        />
        <SettingRow
          title="Default knowledge base for searches"
          desc="Which collection the Evidence Explorer searches unless you override it per query."
          control={
            <select className="select" defaultValue="general" aria-label="Default knowledge base">
              <option value="general">General clinical KB</option>
              <option value="cbt">CBT</option>
              <option value="act">ACT</option>
              <option value="humanistic">Humanistic</option>
            </select>
          }
        />
        <SettingRow
          title="Results per search"
          desc="Number of evidence items returned per query (1–20)."
          control={
            <select className="select" defaultValue="5" aria-label="Results per search">
              {[3, 5, 8, 10, 15, 20].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          }
        />
      </Panel>

      <Panel title="LLM & retrieval">
        <SettingRow
          title="Retrieval-first answers"
          desc="Evidence screens never call the LLM. Generation is reserved for the chat API and planned synthesis features."
          control={<Badge tone="primary" dot>Retrieval-only</Badge>}
        />
        <SettingRow
          title="Hybrid retrieval"
          desc="Dense + BM25 fusion with cross-encoder reranking for framework literature."
          control={<Switch checked onChange={() => {}} label="Hybrid retrieval" />}
        />
      </Panel>

      <Panel title="About">
        <SettingRow
          title="Version"
          desc="Clinical Psychology Assistant — evidence-based clinical support for licensed psychologists."
          control={<Badge tone="neutral">v0.1.0</Badge>}
        />
        <SettingRow
          title="Privacy"
          desc="Case data and uploaded documents remain local. The assistant never generates diagnoses."
          control={<Badge tone="success" dot>Local-first</Badge>}
        />
      </Panel>
    </div>
  );
}

function SettingRow({
  title,
  desc,
  control,
}: {
  title: string;
  desc: string;
  control: React.ReactNode;
}) {
  return (
    <div className="setting-row">
      <div className="setting-info">
        <div className="setting-title">{title}</div>
        <div className="setting-desc">{desc}</div>
      </div>
      {control}
    </div>
  );
}
