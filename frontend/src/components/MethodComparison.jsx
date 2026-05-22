import React, { useMemo, useState } from 'react'
import AnswerCriticPanel from './AnswerCriticPanel.jsx'

function scoreFor(result) {
  if (!result) return '-'
  if (typeof result.hybrid_score === 'number') return result.hybrid_score.toFixed(3)
  if (typeof result.reranker_score === 'number') return result.reranker_score.toFixed(3)
  if (typeof result.agent_score === 'number') return result.agent_score.toFixed(3)
  if (typeof result.multihop_score === 'number') return result.multihop_score.toFixed(3)
  if (typeof result.similarity_score === 'number') return result.similarity_score.toFixed(3)
  if (typeof result.bm25_score === 'number') return result.bm25_score.toFixed(3)
  return '-'
}

function methodVisual(method, payload) {
  const vis = payload?.visData || {}
  const result = payload?.queryResult || {}
  const rows = result.results || []

  if (method === 'naive') {
    const retrieved = rows.length
    const pages = new Set(rows.map((row) => row.page_number).filter(Boolean)).size
    return (
      <div className="comparison-mini-flow">
        <div><span>Document</span><strong>{payload?.chunks?.length || 0}</strong></div>
        <div><span>Retrieved</span><strong>{retrieved}</strong></div>
        <div><span>Pages</span><strong>{pages}</strong></div>
      </div>
    )
  }

  if (method === 'bm25') {
    const terms = vis.query_terms || result.query_terms || []
    const missing = vis.missing_query_terms || result.missing_query_terms || []
    return (
      <div className="comparison-term-strip">
        {terms.slice(0, 8).map((term) => (
          <span key={term} className={missing.includes(term) ? 'missing' : ''}>{term}</span>
        ))}
      </div>
    )
  }

  if (method === 'rerank') {
    const summary = vis.summary || {}
    return (
      <div className="comparison-overlap">
        <div><span>Candidates</span><strong>{summary.candidate_count || result.candidate_count || 0}</strong></div>
        <div><span>Promoted</span><strong>{summary.promoted_count || 0}</strong></div>
        <div><span>Demoted</span><strong>{summary.demoted_count || 0}</strong></div>
      </div>
    )
  }

  if (method === 'graph') {
    const stats = result.graph_stats || {}
    return (
      <div className="comparison-overlap">
        <div><span>Nodes</span><strong>{stats.node_count || 0}</strong></div>
        <div><span>Edges</span><strong>{stats.edge_count || 0}</strong></div>
        <div><span>Used</span><strong>{stats.used_node_count || 0}</strong></div>
      </div>
    )
  }

  if (method === 'vectorless') {
    const stats = result.tree_stats || {}
    return (
      <div className="comparison-overlap">
        <div><span>Sections</span><strong>{stats.section_count || 0}</strong></div>
        <div><span>Paragraphs</span><strong>{stats.paragraph_count || 0}</strong></div>
        <div><span>Path</span><strong>{stats.selected_depth || 0}</strong></div>
      </div>
    )
  }

  if (method === 'agentic') {
    const summary = result.agent_summary || {}
    return (
      <div className="comparison-overlap">
        <div><span>Tools</span><strong>{summary.tool_call_count || 0}</strong></div>
        <div><span>Accepted</span><strong>{summary.accepted_count || rows.length}</strong></div>
        <div><span>Rejected</span><strong>{summary.rejected_count || 0}</strong></div>
      </div>
    )
  }

  if (method === 'multihop') {
    const summary = result.multihop_summary || {}
    return (
      <div className="comparison-overlap">
        <div><span>Hops</span><strong>{summary.hop_count || result.hop_count || 0}</strong></div>
        <div><span>Bridge</span><strong>{summary.confirmed_count || 0}</strong></div>
        <div><span>Terms</span><strong>{(result.bridge_terms || []).length}</strong></div>
      </div>
    )
  }

  const overlap = vis.overlap || result.overlap || {}
  return (
    <div className="comparison-overlap">
      <div><span>Vector only</span><strong>{overlap.vector_only || 0}</strong></div>
      <div><span>Both</span><strong>{overlap.both || 0}</strong></div>
      <div><span>BM25 only</span><strong>{overlap.bm25_only || 0}</strong></div>
    </div>
  )
}

const METHOD_COLORS = ['#f59e0b', '#0ea5e9', '#10b981', '#ef4444', '#7c3aed', '#f97316', '#ec4899', '#06b6d4']

export default function MethodComparison({ methods, comparisonData, evalData }) {
  const [activeMetrics, setActiveMetrics] = useState(true)

  const matrix = useMemo(() => {
    const idsByMethod = new Map()
    const pages = new Map()
    methods.forEach((method) => {
      const results = comparisonData[method.id]?.queryResult?.results || []
      idsByMethod.set(method.id, new Set(results.map((row) => row.chunk_id).filter(Boolean)))
      results.forEach((row) => {
        if (!row.page_number) return
        const entry = pages.get(row.page_number) || new Set()
        entry.add(method.shortLabel)
        pages.set(row.page_number, entry)
      })
    })
    const overlapRows = methods.map((left) => methods.map((right) => {
      const a = idsByMethod.get(left.id) || new Set()
      const b = idsByMethod.get(right.id) || new Set()
      return Array.from(a).filter((id) => b.has(id)).length
    }))
    return {
      overlapRows,
      pages: Array.from(pages.entries())
        .map(([page, set]) => ({ page, methods: Array.from(set) }))
        .sort((left, right) => Number(left.page) - Number(right.page))
        .slice(0, 40)
    }
  }, [methods, comparisonData])

  const ranking = evalData?.comparison?.ranking?.['ndcg@5'] || []
  const bestMethod = ranking[0]?.method || null

  return (
    <section className="panel comparison-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3>All-Method Comparison</h3>
        <button
          type="button"
          className="mode-pill active"
          onClick={() => setActiveMetrics(!activeMetrics)}
          style={{ fontSize: 11 }}
        >
          {activeMetrics ? 'Show Metrics' : 'Show Cards'}
        </button>
      </div>

      {activeMetrics && evalData?.comparison?.table && (
        <div className="viz-card">
          <h4>Performance Leaderboard</h4>
          <div className="ranking-leaderboard">
            {ranking.map((entry, i) => (
              <div key={entry.method} className="ranking-row" style={{ borderColor: METHOD_COLORS[i] + '40' }}>
                <span className={`ranking-rank ${i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : ''}`}>
                  {i + 1}
                </span>
                <span style={{ color: METHOD_COLORS[i], fontWeight: 700, fontSize: 12 }}>
                  {methods.find(m => m.id === entry.method)?.label || entry.method}
                </span>
                <strong style={{ fontSize: 14, color: METHOD_COLORS[i] }}>{entry.score.toFixed(3)}</strong>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="comparison-dashboard">
        <article className="viz-card">
          <h4>Evidence Overlap Matrix</h4>
          <div className="comparison-matrix" style={{ '--method-count': methods.length }}>
            <span />
            {methods.map((method) => <strong key={method.id}>{method.shortLabel}</strong>)}
            {matrix.overlapRows.map((row, rowIndex) => (
              <React.Fragment key={methods[rowIndex].id}>
                <strong>{methods[rowIndex].shortLabel}</strong>
                {row.map((count, colIndex) => (
                  <i key={`${rowIndex}-${colIndex}`} style={{ opacity: Math.min(1, 0.14 + count * 0.16) }}>{count}</i>
                ))}
              </React.Fragment>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Citation Coverage Map</h4>
          <div className="comparison-page-map">
            {matrix.pages.length ? matrix.pages.map((entry) => (
              <div key={entry.page} style={{ '--coverage': entry.methods.length / Math.max(1, methods.length) }}>
                <strong>P{entry.page}</strong>
                <span>{entry.methods.join(', ')}</span>
              </div>
            )) : <div className="comparison-empty">No citation pages loaded yet.</div>}
          </div>
        </article>
      </div>

      <article className="viz-card comparison-wide-table-card">
        <h4>Answer and Evidence Table</h4>
        <div className="comparison-wide-table">
          <div className="comparison-wide-row comparison-wide-head">
            <span>Method</span>
            <span>Mode</span>
            <span>Top page</span>
            <span>Score</span>
            <span>Answer</span>
          </div>
          {methods.map((method) => {
            const payload = comparisonData[method.id]
            const queryResult = payload?.queryResult
            const topResult = queryResult?.results?.[0]
            return (
              <div key={`${method.id}-wide`} className="comparison-wide-row">
                <strong style={{ color: METHOD_COLORS[methods.indexOf(method) % METHOD_COLORS.length] }}>
                  {method.shortLabel}
                </strong>
                <span>{method.evidenceMode}</span>
                <span>P{topResult?.page_number || '-'}</span>
                <span>{scoreFor(topResult)}</span>
                <p>{queryResult?.answer || topResult?.chunk_text_preview || payload?.error || 'No answer loaded yet.'}</p>
              </div>
            )
          })}
        </div>
      </article>

      <div className="comparison-grid">
        {methods.map((method) => {
          const payload = comparisonData[method.id]
          const queryResult = payload?.queryResult
          const results = queryResult?.results || []
          const topResult = results[0]

          return (
            <article key={method.id} className="comparison-card">
              <div className="comparison-card-head">
                <h4>{method.label}</h4>
                <span>{method.evidenceMode}</span>
              </div>

              {payload?.error ? (
                <div className="comparison-empty">{payload.error}</div>
              ) : (
                <>
                  {methodVisual(method.id, payload)}

                  <AnswerCriticPanel queryResult={queryResult} title="Answer" />

                  <div className="comparison-best">
                    <h5>Top Evidence</h5>
                    <div className="comparison-evidence-row">
                      <span>Page {topResult?.page_number || '-'}</span>
                      <strong>{scoreFor(topResult)}</strong>
                    </div>
                    <p>{topResult?.chunk_text_preview || 'No retrieved evidence yet.'}</p>
                  </div>
                </>
              )}
            </article>
          )
        })}
      </div>
    </section>
  )
}
