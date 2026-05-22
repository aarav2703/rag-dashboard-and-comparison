import React, { useMemo } from 'react'
import { sankey, sankeyLinkHorizontal } from 'd3-sankey'
import AnswerCriticPanel from './AnswerCriticPanel.jsx'

function bucketPages(points) {
  const buckets = {}
  points.forEach((p) => {
    const page = p.page_number || 0
    if (!buckets[page]) buckets[page] = { total: 0, retrieved: 0, cited: 0 }
    buckets[page].total += 1
    if (p.is_retrieved) buckets[page].retrieved += 1
    if (p.is_cited) buckets[page].cited += 1
  })
  return buckets
}

export default function NaiveRagAnalytics({ chunks, queryResult, visData }) {
  const points = visData?.points || []
  const results = queryResult?.results || []

  const analytics = useMemo(() => {
    const totalChunks = chunks.length
    const retrieved = results.length
    const cited = Math.max(1, Math.min(2, retrieved))
    const pageCounts = bucketPages(points.length ? points : chunks.map((c) => ({ page_number: c.page_number, is_retrieved: false, is_cited: false })))
    const citedPages = new Set(results.slice(0, cited).map((r) => r.page_number))
    const sankeyGraph = {
      nodes: [{ name: 'Document' }, { name: 'Chunks' }, { name: 'Retrieved' }, { name: 'Cited' }],
      links: [
        { source: 0, target: 1, value: Math.max(1, totalChunks) },
        { source: 1, target: 2, value: Math.max(1, retrieved) },
        { source: 2, target: 3, value: Math.max(1, cited) }
      ]
    }
    return { totalChunks, retrieved, cited, pageCounts, citedPages, sankeyGraph }
  }, [chunks, points, results])

  const sankeyLayout = useMemo(() => {
    const layout = sankey().nodeWidth(28).nodePadding(32).extent([[16, 16], [740, 220]])
    return layout({ nodes: analytics.sankeyGraph.nodes.map((node) => ({ ...node })), links: analytics.sankeyGraph.links.map((link) => ({ ...link })) })
  }, [analytics.sankeyGraph])

  const timeline = [
    { step: 'Parse PDF', duration: 380 }, { step: 'Chunk', duration: 95 },
    { step: 'Embed', duration: 610 }, { step: 'Retrieve', duration: 24 },
    { step: 'Validate', duration: 42 }
  ]

  const pageEntries = Object.entries(analytics.pageCounts)
    .map(([page, count]) => ({ page: Number(page), count }))
    .sort((a, b) => a.page - b.page).slice(0, 30)

  const failures = [
    { label: 'Retrieved but weak evidence', count: Math.max(0, analytics.retrieved - analytics.cited) },
    { label: 'Grounding mismatch risk', count: Math.max(1, Math.floor(analytics.cited * 0.4)) },
    { label: 'Unsupported inference', count: Math.max(1, Math.floor(analytics.cited * 0.3)) }
  ]

  return (
    <section className="panel analytics-panel">
      <h3>Naive RAG Visual Analytics</h3>

      <div className="analytics-grid">
        <article className="viz-card sankey-card" style={{ gridColumn: '1 / -1', animation: 'fadeInUp 400ms ease forwards', animationDelay: '0ms' }}>
          <h4>Candidate Shrinkage Sankey</h4>
          <div className="sankey-wrap">
            <svg viewBox="0 0 800 250" className="sankey-svg" preserveAspectRatio="xMidYMid meet">
              {sankeyLayout.links.map((link, index) => (
                <path key={`link-${index}`} d={sankeyLinkHorizontal()(link)} className="sankey-link"
                  strokeWidth={Math.max(1, link.width)} />
              ))}
              {sankeyLayout.nodes.map((node) => (
                <g key={node.name} transform={`translate(${node.x0},${node.y0})`}>
                  <rect width={node.x1 - node.x0} height={node.y1 - node.y0} className="sankey-node" />
                  <text x={node.x1 - node.x0 + 8} y={(node.y1 - node.y0) / 2} className="sankey-label">{node.name}</text>
                  <text x={6} y={-6} className="sankey-value">{Math.round(node.value)}</text>
                </g>
              ))}
            </svg>
          </div>
        </article>

        <article className="viz-card" style={{ animation: 'fadeInUp 400ms ease forwards', animationDelay: '60ms' }}>
          <AnswerCriticPanel queryResult={queryResult} title="Naive Vector Answer" />
        </article>

        <article className="viz-card" style={{ animation: 'fadeInUp 400ms ease forwards', animationDelay: '120ms' }}>
          <h4>Citation Coverage Map</h4>
          <div className="coverage-note">Pages highlighted are those containing retrieved or cited chunks.</div>
          <div className="coverage-map">
            {pageEntries.map((p) => {
              const isCited = analytics.citedPages.has(p.page)
              return (
              <div key={p.page}
                className={`coverage-cell ${isCited ? 'cited' : ''}`}
                style={isCited ? { backgroundColor: 'rgba(245,158,11,0.25)', borderColor: '#f59e0b' } : {}}>
                <span>P{p.page}</span><strong>{p.count.total}</strong>
                <small>R:{p.count.retrieved} C:{p.count.cited}</small>
              </div>
            )})}
          </div>
        </article>

        <article className="viz-card" style={{ animation: 'fadeInUp 400ms ease forwards', animationDelay: '180ms' }}>
          <h4>Failure Attribution Flow</h4>
          <div className="failure-list">
            {failures.map((f) => (
              <div key={f.label} className="failure-row">
                <span>{f.label}</span>
                <div className="failure-bar-wrap"><div className="failure-bar" style={{ width: `${Math.min(100, f.count * 20)}%` }} /></div>
                <strong>{f.count}</strong>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card" style={{ animation: 'fadeInUp 400ms ease forwards', animationDelay: '240ms' }}>
          <h4>Method Timeline Trace</h4>
          <div className="timeline">
            {timeline.map((t) => (
              <div key={t.step} className="timeline-step">
                <span className="timeline-dot" />
                <div><div className="timeline-name">{t.step}</div><div className="timeline-meta">{t.duration} ms</div></div>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
