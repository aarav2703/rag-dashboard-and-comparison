import React, { useEffect, useMemo, useState } from 'react'

function nodeClass(node) {
  return `agent-flow-node ${node.type || 'tool'} ${node.status || 'complete'}`
}

function scoreLabel(value) {
  return typeof value === 'number' ? value.toFixed(3) : value || '-'
}

const TOOL_NODES = [
  ['vector', 'Vector'],
  ['hybrid', 'Hybrid'],
  ['graph', 'Graph'],
  ['rerank', 'Rerank'],
  ['second_hop', 'Hop 2'],
  ['web_search', 'Web'],
  ['answer', 'Answer']
]

export default function AgenticRagAnalytics({ queryResult, visData }) {
  const [selectedStep, setSelectedStep] = useState(null)
  const [replayIndex, setReplayIndex] = useState(0)
  const [replaying, setReplaying] = useState(false)
  const results = queryResult?.results || []
  const flow = visData?.control_flow || { nodes: [], links: [] }
  const timeline = visData?.tool_timeline || queryResult?.tool_timeline || []
  const scratchpad = visData?.scratchpad || queryResult?.scratchpad || []
  const rejected = visData?.rejected_evidence || queryResult?.rejected_evidence || []
  const acceptedPath = visData?.accepted_path || queryResult?.accepted_path || []
  const summary = queryResult?.agent_summary || {}
  const plannerDecisions = visData?.planner_decisions || queryResult?.planner_decisions || []
  const toolExecutionTrace = visData?.tool_execution_trace || queryResult?.tool_execution_trace || []
  const width = 860
  const height = 340
  const maxDuration = Math.max(...timeline.map((t) => t.duration_ms || 0), 100)
  const decisionRows = (toolExecutionTrace.length ? toolExecutionTrace : plannerDecisions).map((row, index) => ({
    step: row.step || index + 1,
    decision: row.decision || row.tool || 'Tool decision',
    tool: row.tool || 'planner',
    query: row.query || queryResult?.query || '',
    result_count: row.result_count ?? 0,
    confidence: row.confidence,
    next_action: row.next_action || row.status || 'continue'
  }))
  const usedTools = new Set([
    ...timeline.map((row) => row.tool),
    ...plannerDecisions.map((row) => row.tool),
    ...toolExecutionTrace.map((row) => row.tool),
    ...(results || []).map((row) => row.source)
  ].filter(Boolean))

  const layout = useMemo(() => {
    const nodes = flow.nodes.map((node, index) => ({
      ...node, x: 70 + index * ((width - 140) / Math.max(1, flow.nodes.length - 1)), y: index % 2 ? 198 : 118
    }))
    const byId = new Map(nodes.map((node) => [node.id, node]))
    const links = flow.links.map((link) => ({ ...link, sourceNode: byId.get(link.source), targetNode: byId.get(link.target) })).filter((link) => link.sourceNode && link.targetNode)
    return { nodes, links }
  }, [flow])

  useEffect(() => {
    if (!replaying) return
    if (replayIndex >= scratchpad.length) { setReplaying(false); return }
    const timer = setTimeout(() => setReplayIndex((prev) => prev + 1), 600)
    return () => clearTimeout(timer)
  }, [replaying, replayIndex, scratchpad.length])

  function startReplay() {
    setReplayIndex(0)
    setReplaying(true)
  }

  if (!flow.nodes.length && !results.length) {
    return (
      <section className="panel agentic-panel">
        <h3>Agentic RAG Control Flow</h3>
        <div className="visual-empty"><strong>No agent trace yet</strong><span>Run Agentic RAG and ask a question to see planning, tool calls, critique, retries, and final evidence.</span></div>
      </section>
    )
  }

  return (
    <section className="panel agentic-panel">
      <div className="agentic-head">
        <div><h3>Agentic RAG Control Flow</h3><p className="coverage-note">Agentic RAG is orchestration: the useful visual is the decision trace, not just the final chunks.</p></div>
        <div className="agentic-stats">
          <span>{summary.primary_tool || 'planner'} route</span>
          <span>{scoreLabel(summary.planner_confidence)} confidence</span>
          <span>{summary.tool_call_count || timeline.length} tools</span>
          <span>{summary.retry_used ? 'retry used' : 'no retry'}</span>
        </div>
      </div>

      <div className="agentic-layout">
        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <h4>Agentic Multi-hop Answer</h4>
          <p className="answer-critic-text">{queryResult?.answer || results[0]?.chunk_text_preview || 'No answer available yet.'}</p>
          <div className="answer-critic-meta">
            <span>Evidence {queryResult?.evidence_count ?? results.length}</span>
            <span>{queryResult?.answer_source || 'awaiting answer'}</span>
            <span>{(queryResult?.bridge_terms || []).length} bridge terms</span>
          </div>
        </article>

        <article className="viz-card agent-decision-board-card" style={{ gridColumn: '1 / -1' }}>
          <h4>Agent Decision Loop Board</h4>
          <div className="agent-decision-board">
            <div className="agent-decision-head">
              <span>Step</span><span>Decision</span><span>Tool</span><span>Query</span><span>Results</span><span>Confidence</span><span>Next</span>
            </div>
            {decisionRows.length ? decisionRows.slice(0, 8).map((row) => (
              <div key={`${row.step}-${row.tool}-${row.next_action}`} className="agent-decision-row">
                <span>{row.step}</span>
                <strong>{row.decision}</strong>
                <em>{String(row.tool).replace('_', ' ')}</em>
                <p>{row.query}</p>
                <span>{row.result_count}</span>
                <div className="agent-confidence-meter"><i style={{ width: `${Math.max(4, Math.min(100, Number(row.confidence || 0) * 100))}%` }} /></div>
                <span>{row.next_action}</span>
              </div>
            )) : <div className="comparison-empty">No structured decision rows returned.</div>}
          </div>
        </article>

        <article className="viz-card agent-tool-network-card" style={{ gridColumn: '1 / -1' }}>
          <h4>Tool Network Map</h4>
          <svg viewBox="0 0 760 260" className="agent-tool-network-svg" preserveAspectRatio="xMidYMid meet">
            <circle cx="380" cy="130" r="42" className="agent-tool-planner" />
            <text x="380" y="126" textAnchor="middle">Planner</text>
            <text x="380" y="141" textAnchor="middle">{scoreLabel(summary.planner_confidence)}</text>
            {TOOL_NODES.map(([id, label], index) => {
              const angle = (index / TOOL_NODES.length) * Math.PI * 2 - Math.PI / 2
              const x = 380 + Math.cos(angle) * 230
              const y = 130 + Math.sin(angle) * 88
              const active = usedTools.has(id) || usedTools.has(label.toLowerCase()) || (id === 'second_hop' && usedTools.has('hop2'))
              return (
                <g key={id} className={`agent-tool-node ${active ? 'active' : 'skipped'}`}>
                  <line x1="380" y1="130" x2={x} y2={y} />
                  <circle cx={x} cy={y} r="28" />
                  <text x={x} y={y + 4} textAnchor="middle">{label}</text>
                </g>
              )
            })}
          </svg>
        </article>

        <article className="viz-card agentic-flow-card">
          <h4>Tool-Call Control Flow</h4>
          <svg className="agent-flow-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
            <rect x="0" y="0" width={width} height={height} className="agent-flow-bg" />
            {layout.links.map((link, index) => (
              <path key={`${link.source}-${link.target}-${index}`}
                d={`M${link.sourceNode.x + 34},${link.sourceNode.y} C${(link.sourceNode.x + link.targetNode.x) / 2},${link.sourceNode.y} ${(link.sourceNode.x + link.targetNode.x) / 2},${link.targetNode.y} ${link.targetNode.x - 34},${link.targetNode.y}`}
                className="agent-flow-link" />
            ))}
            {layout.nodes.map((node) => {
              const isActive = selectedStep === node.id
              return (
              <g key={node.id} className={nodeClass(node)}
                transform={`translate(${node.x},${node.y})`}
                onMouseEnter={() => setSelectedStep(node.id)}
                onMouseLeave={() => setSelectedStep(null)}>
                <circle r="34"
                  strokeWidth={isActive ? 3.5 : 2}
                  style={{ transition: 'stroke-width 160ms ease' }} />
                <text y="-4">{node.label}</text>
                <text y="13" className="agent-flow-status">{node.status}</text>
              </g>
            )})}
          </svg>
        </article>

        <article className="viz-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4>Agent Scratchpad Trace</h4>
            <button type="button" onClick={startReplay} className="mode-pill"
              style={{ fontSize: 10, padding: '5px 10px' }}>{replaying ? 'Replaying...' : 'Replay'}</button>
          </div>
          <div className="agent-scratchpad">
            {scratchpad.map((line, index) => (
              <div key={`${line}-${index}`} className={index < replayIndex ? 'active' : ''}
                style={{ opacity: index < replayIndex ? 1 : 0.5, transition: 'opacity 300ms ease' }}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <p>{line}</p>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Tool-Use Gantt Timeline</h4>
          <div className="agent-timeline">
            {timeline.map((step) => {
              const barPct = Math.max(6, ((step.duration_ms || 1) / maxDuration) * 100)
              return (
                <div key={`${step.step}-${step.tool}`} className={`agent-timeline-row ${step.status}`}>
                  <span>{step.step}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <strong style={{ fontSize: 11 }}>{step.label}</strong>
                      <small>{step.tool} | {step.duration_ms || 0} ms</small>
                    </div>
                    <div className="bm25-bar-track" style={{ height: 8 }}>
                      <div className="bm25-bar-fill bm25-bar-fill-accent" style={{ width: `${barPct}%`, transition: 'width 600ms ease' }} />
                    </div>
                  </div>
                  <em>{step.status}</em>
                </div>
              )
            })}
          </div>
        </article>

        <article className="viz-card">
          <h4>Planner Decisions</h4>
          <div className="agent-timeline">
            {plannerDecisions.length ? plannerDecisions.map((step, index) => (
              <div key={`${step.tool}-${step.decision}-${index}`} className="agent-timeline-row complete">
                <span>{step.step || index + 1}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <strong style={{ fontSize: 11 }}>{step.decision || step.tool}</strong>
                    <small>{step.tool} | {scoreLabel(step.confidence)} | {step.result_count ?? 0} results</small>
                  </div>
                  <p style={{ margin: 0, color: 'var(--muted)', fontSize: 11 }}>{step.reason_summary || step.next_action}</p>
                  {step.error && <p style={{ margin: '4px 0 0', color: 'var(--danger)', fontSize: 11 }}>{step.error}</p>}
                </div>
                <em>{step.next_action || 'continue'}</em>
              </div>
            )) : <div className="comparison-empty">No planner decisions returned for this query.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Executed Tool Loop</h4>
          <div className="agent-timeline">
            {toolExecutionTrace.length ? toolExecutionTrace.map((step, index) => (
              <div key={`${step.tool}-${step.status}-${index}`} className={`agent-timeline-row ${step.status || 'complete'}`}>
                <span>{index + 1}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <strong style={{ fontSize: 11 }}>{String(step.tool || 'tool').replace('_', ' ')}</strong>
                    <small>{scoreLabel(step.confidence)} confidence</small>
                  </div>
                  <p style={{ margin: 0, color: 'var(--muted)', fontSize: 11 }}>{step.query || step.reason_summary}</p>
                </div>
                <em>{step.result_count ?? 0} new</em>
              </div>
            )) : <div className="comparison-empty">No extra tool loop was needed for this query.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Final Accepted Evidence</h4>
          <div className="agent-accepted-path">
            {acceptedPath.map((item, index) => (
              <React.Fragment key={`${item.step}-${item.label}-${index}`}>
                <span className={item.step}>{item.label}</span>
                {index < acceptedPath.length - 1 && <i />}
              </React.Fragment>
            ))}
          </div>
          <div className="agent-accepted-list">
            {results.map((result) => (
              <div key={result.chunk_id}>
                <strong>#{result.rank} P{result.page_number} | {scoreLabel(result.agent_score)}</strong>
                <small>{result.source} | {result.accepted_reason}</small>
                {result.url && <small>{result.title || result.url}</small>}
                <p>{result.chunk_text_preview}</p>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Rejected Evidence Pile</h4>
          <div className="agent-rejected-pile">
            {rejected.length ? rejected.map((item) => (
              <div key={item.chunk_id}>
                <strong>P{item.page_number} | {scoreLabel(item.agent_score)}</strong>
                <span>{item.reason}</span>
                <p>{item.chunk_text_preview}</p>
              </div>
            )) : <div className="comparison-empty">No rejected evidence for this query.</div>}
          </div>
        </article>
      </div>
    </section>
  )
}
