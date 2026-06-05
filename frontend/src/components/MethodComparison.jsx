import React, { useMemo, useState } from 'react'
import { arc, chord, ribbon } from 'd3'
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

  if (method === 'crag') {
    const summary = result.crag_summary || {}
    const rerankSummary = vis.summary || {}
    return (
      <div className="comparison-overlap">
        <div><span>Branch</span><strong>{summary.branch || '-'}</strong></div>
        <div><span>Promoted</span><strong>{rerankSummary.promoted_count || 0}</strong></div>
        <div><span>Fallback</span><strong>{summary.fallback_count || 0}</strong></div>
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

  if (method === 'agentic') {
    const summary = result.agent_summary || {}
    return (
      <div className="comparison-overlap">
        <div><span>Tools</span><strong>{summary.tool_call_count || 0}</strong></div>
        <div><span>Hops</span><strong>{summary.hop_count || 0}</strong></div>
        <div><span>Bridge</span><strong>{(summary.bridge_terms || result.bridge_terms || []).length}</strong></div>
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

function previewFor(row) {
  return row?.chunk_text_preview || row?.preview || row?.title || row?.url || 'No preview available.'
}

function AgreementChord({ methods, matrix, uniqueCounts, totalCounts }) {
  const width = 480
  const height = 420
  const outerRadius = 166
  const innerRadius = 146
  const layout = chord().padAngle(0.045).sortSubgroups((a, b) => b - a)(matrix)
  const arcPath = arc().innerRadius(innerRadius).outerRadius(outerRadius)
  const ribbonPath = ribbon().radius(innerRadius - 8)
  const hasLinks = matrix.some((row, rowIndex) => row.some((value, colIndex) => rowIndex !== colIndex && value > 0))

  return (
    <div className="agreement-chord-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="agreement-chord-svg" role="img" aria-label="Evidence agreement chord diagram">
        <g transform={`translate(${width / 2},${height / 2})`}>
          <circle r={innerRadius - 18} className="agreement-chord-core" />
          {layout.groups.map((group) => {
            const method = methods[group.index]
            const angle = (group.startAngle + group.endAngle) / 2
            const labelRadius = outerRadius + 26
            const x = Math.cos(angle - Math.PI / 2) * labelRadius
            const y = Math.sin(angle - Math.PI / 2) * labelRadius
            return (
              <g key={method.id}>
                <path d={arcPath(group)} fill={METHOD_COLORS[group.index % METHOD_COLORS.length]} className="agreement-arc">
                  <title>{`${method.label}: ${uniqueCounts[method.id] || 0} unique / ${totalCounts[method.id] || 0} retrieved`}</title>
                </path>
                <text x={x} y={y} textAnchor="middle" className="agreement-label">{method.shortLabel}</text>
                <text x={x} y={y + 13} textAnchor="middle" className="agreement-sub">{uniqueCounts[method.id] || 0} unique</text>
              </g>
            )
          })}
          {layout.map((link, index) => (
            <path
              key={`${link.source.index}-${link.target.index}-${index}`}
              d={ribbonPath(link)}
              fill={METHOD_COLORS[link.source.index % METHOD_COLORS.length]}
              className="agreement-ribbon"
              style={{ opacity: Math.min(0.72, 0.14 + (link.source.value || 0) * 0.09) }}
            >
              <title>{`${methods[link.source.index]?.shortLabel} <-> ${methods[link.target.index]?.shortLabel}: ${link.source.value} shared chunks`}</title>
            </path>
          ))}
          {!hasLinks && <text textAnchor="middle" className="agreement-empty">No shared evidence yet</text>}
        </g>
      </svg>
      <div className="agreement-legend">
        {methods.map((method, index) => (
          <span key={method.id}><i style={{ background: METHOD_COLORS[index % METHOD_COLORS.length] }} />{method.shortLabel}</span>
        ))}
      </div>
    </div>
  )
}

function ConsensusHeatmap({ methods, rows }) {
  if (!rows.length) return <div className="comparison-empty">No evidence consensus rows loaded yet.</div>
  return (
    <div className="consensus-heatmap" style={{ '--method-count': methods.length }}>
      <div className="consensus-head"><span>Evidence</span>{methods.map((method) => <strong key={method.id}>{method.shortLabel}</strong>)}</div>
      {rows.slice(0, 28).map((row) => (
        <div key={row.chunkId} className="consensus-row">
          <span title={row.preview}>P{row.page || '-'} · {row.preview}</span>
          {methods.map((method) => {
            const hit = row.methods[method.id]
            const strength = hit ? Math.max(0.16, 1 - ((hit.rank || 10) - 1) / 10) : 0
            return (
              <i key={method.id} style={{ '--strength': strength }} title={hit ? `${method.label}: rank ${hit.rank || '-'} | ${scoreFor(hit)}` : `${method.label}: not retrieved`}>
                {hit?.rank ? `#${hit.rank}` : ''}
              </i>
            )
          })}
        </div>
      ))}
    </div>
  )
}

export default function MethodComparison({ methods, comparisonData, evalData }) {
  const [activeMetrics, setActiveMetrics] = useState(true)

  const matrix = useMemo(() => {
    const idsByMethod = new Map()
    const rowsByMethod = new Map()
    const evidenceRows = new Map()
    const pages = new Map()
    methods.forEach((method) => {
      const results = comparisonData[method.id]?.queryResult?.results || []
      rowsByMethod.set(method.id, results)
      idsByMethod.set(method.id, new Set(results.map((row) => row.chunk_id).filter(Boolean)))
      results.forEach((row) => {
        if (row.chunk_id) {
          const current = evidenceRows.get(row.chunk_id) || {
            chunkId: row.chunk_id,
            page: row.page_number,
            preview: previewFor(row),
            methods: {}
          }
          current.methods[method.id] = row
          evidenceRows.set(row.chunk_id, current)
        }
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
    const uniqueCounts = {}
    const totalCounts = {}
    methods.forEach((method) => {
      const own = idsByMethod.get(method.id) || new Set()
      totalCounts[method.id] = own.size
      uniqueCounts[method.id] = Array.from(own).filter((id) => methods.every((other) => other.id === method.id || !(idsByMethod.get(other.id) || new Set()).has(id))).length
    })
    return {
      overlapRows,
      uniqueCounts,
      totalCounts,
      evidenceRows: Array.from(evidenceRows.values())
        .sort((left, right) => Object.keys(right.methods).length - Object.keys(left.methods).length)
        .slice(0, 80),
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
        <article className="viz-card comparison-chord-card">
          <h4>Evidence Agreement Chord</h4>
          <AgreementChord methods={methods} matrix={matrix.overlapRows} uniqueCounts={matrix.uniqueCounts} totalCounts={matrix.totalCounts} />
        </article>

        <article className="viz-card">
          <h4>Evidence Consensus Heatmap</h4>
          <ConsensusHeatmap methods={methods} rows={matrix.evidenceRows} />
        </article>
      </div>

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

                  {method.id === 'crag' ? (
                    <AnswerCriticPanel queryResult={queryResult} title="CRAG Answer" />
                  ) : (
                    <div className="answer-critic-panel">
                      <div className="answer-critic-head">
                        <div><h4>Answer</h4><span>{queryResult?.answer_source || 'awaiting answer'}</span></div>
                        <strong className="critic-verdict accepted">{queryResult?.answer ? 'Ready' : 'Pending'}</strong>
                      </div>
                      <p className="answer-critic-text">{queryResult?.answer || topResult?.chunk_text_preview || 'No answer loaded yet.'}</p>
                    </div>
                  )}

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
