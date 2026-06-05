import React, { useMemo, useState } from 'react'
import { sankey, sankeyLinkHorizontal } from 'd3-sankey'

function sourceLabel(source) {
  if (source === 'vector-only') return 'Vector only'
  if (source === 'bm25-only') return 'BM25 only'
  return 'Both'
}

function rankY(rank, maxRank = 10) {
  if (!rank) return 256
  return 42 + ((Math.min(rank, maxRank) - 1) / Math.max(1, maxRank - 1)) * 190
}

export default function HybridAnalytics({ queryResult, visData }) {
  const [sourceFilter, setSourceFilter] = useState('all')
  const [hoverRank, setHoverRank] = useState(null)
  const results = queryResult?.results || []
  const rankFusionTable = queryResult?.rank_fusion_table || visData?.rank_fusion_table || []
  const overlap = visData?.overlap || queryResult?.overlap || {}
  const mergeSankey = visData?.merge_sankey || {
    nodes: [{ name: 'Vector candidates' }, { name: 'BM25 candidates' }, { name: 'Merged candidates' }, { name: 'Final evidence' }],
    links: [
      { source: 0, target: 2, value: overlap.vector_candidates || 1 },
      { source: 1, target: 2, value: overlap.bm25_candidates || 1 },
      { source: 2, target: 3, value: results.length || 1 }
    ]
  }

  const sankeyLayout = useMemo(() => {
    const layout = sankey().nodeWidth(22).nodePadding(28).extent([[16, 18], [660, 260]])
    return layout({ nodes: mergeSankey.nodes.map((node) => ({ ...node })), links: mergeSankey.links.map((link) => ({ ...link })) })
  }, [mergeSankey])

  const filteredResults = sourceFilter === 'all' ? results : results.filter((result) => result.source === sourceFilter)

  if (!results.length && !visData?.merge_sankey) {
    return (
      <section className="panel hybrid-panel">
        <h3>Hybrid Retrieval Merge</h3>
        <div className="visual-empty"><strong>No hybrid merge data yet</strong><span>Run the Hybrid RAG pipeline to generate vector/BM25 overlap and rank-fusion artifacts.</span></div>
      </section>
    )
  }

  return (
    <section className="panel hybrid-panel">
      <h3>Hybrid Retrieval Merge</h3>
      <div className="hybrid-grid">
        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <h4>Hybrid Answer</h4>
          <p className="answer-critic-text">{queryResult?.answer || results[0]?.chunk_text_preview || 'No answer available yet.'}</p>
          <div className="answer-critic-meta">
            <span>Evidence {queryResult?.evidence_count ?? results.length}</span>
            <span>{queryResult?.answer_source || 'awaiting answer'}</span>
          </div>
        </article>

        <article className="viz-card hybrid-sankey-card">
          <h4>Dual-Source Merge Sankey</h4>
          <div className="coverage-note">Vector and BM25 candidate pools are merged, then rank-fused into final evidence.</div>
          <div className="sankey-wrap">
            <svg viewBox="0 0 700 290" className="sankey-svg" preserveAspectRatio="xMidYMid meet">
              {sankeyLayout.links.map((link, index) => (
                <path key={`hybrid-link-${index}`} d={sankeyLinkHorizontal()(link)} className="hybrid-sankey-link" strokeWidth={Math.max(2, link.width)}
                  style={{ animation: `fadeIn 500ms ease forwards`, animationDelay: `${index * 0.2}s` }} />
              ))}
              {sankeyLayout.nodes.map((node) => (
                <g key={node.name} transform={`translate(${node.x0},${node.y0})`}>
                  <rect width={node.x1 - node.x0} height={node.y1 - node.y0} className="hybrid-sankey-node" />
                  <text x={node.x1 - node.x0 + 8} y={(node.y1 - node.y0) / 2} className="sankey-label">{node.name}</text>
                  <text x={6} y={-6} className="sankey-value">{Math.round(node.value)}</text>
                </g>
              ))}
            </svg>
          </div>
        </article>

        <article className="viz-card">
          <h4>Overlap Matrix</h4>
          <div className="overlap-grid">
            <button type="button" className={`overlap-cell vector-only ${sourceFilter === 'vector-only' ? 'active' : ''}`} onClick={() => setSourceFilter((c) => c === 'vector-only' ? 'all' : 'vector-only')}>
              <span>Vector only</span><strong>{overlap.vector_only || 0}</strong>
            </button>
            <button type="button" className={`overlap-cell both ${sourceFilter === 'both' ? 'active' : ''}`} onClick={() => setSourceFilter((c) => c === 'both' ? 'all' : 'both')}>
              <span>Both</span><strong>{overlap.both || 0}</strong>
            </button>
            <button type="button" className={`overlap-cell bm25-only ${sourceFilter === 'bm25-only' ? 'active' : ''}`} onClick={() => setSourceFilter((c) => c === 'bm25-only' ? 'all' : 'bm25-only')}>
              <span>BM25 only</span><strong>{overlap.bm25_only || 0}</strong>
            </button>
          </div>
        </article>

        <article className="viz-card">
          <h4>Source Badges {sourceFilter !== 'all' ? `(${sourceLabel(sourceFilter)})` : ''}</h4>
          <div className="hybrid-results">
            {filteredResults.map((result, idx) => (
              <div key={result.chunk_id} className="hybrid-result-row" style={{ animation: `fadeInUp 360ms ease forwards`, animationDelay: `${idx * 0.08}s` }}>
                <span className={`source-badge ${result.source}`}>{sourceLabel(result.source)}</span>
                <div><strong>#{result.rank} page {result.page_number}</strong><p>{result.chunk_text_preview}</p></div>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Three-Lane Rank Fusion Chart</h4>
          <div className="fusion-lane-wrap">
            <svg viewBox="0 0 680 300" className="fusion-lane-svg" preserveAspectRatio="xMidYMid meet">
              {[
                ['Vector', 120],
                ['BM25', 340],
                ['Fused', 560]
              ].map(([label, x]) => (
                <g key={label}>
                  <line x1={x} y1="36" x2={x} y2="248" className="fusion-lane-axis" />
                  <text x={x} y="22" textAnchor="middle" className="fusion-lane-title">{label}</text>
                </g>
              ))}
              {rankFusionTable.slice(0, 10).map((row) => {
                const vy = rankY(row.vector_rank)
                const by = rankY(row.bm25_rank)
                const fy = rankY(row.rank)
                const active = hoverRank === row.rank
                return (
                  <g key={`${row.chunk_id}-lane`} className={`fusion-lane-path ${row.source} ${active ? 'active' : ''}`}
                    onMouseEnter={() => setHoverRank(row.rank)} onMouseLeave={() => setHoverRank(null)}>
                    <path d={`M120,${vy} C210,${vy} 250,${by} 340,${by} C430,${by} 470,${fy} 560,${fy}`} />
                    <circle cx="120" cy={vy} r={row.vector_rank ? 5 : 3} />
                    <circle cx="340" cy={by} r={row.bm25_rank ? 5 : 3} />
                    <circle cx="560" cy={fy} r="6" />
                    <text x="586" y={fy + 4}>#{row.rank}</text>
                    <title>{`${sourceLabel(row.source)} | vector ${row.vector_rank || '-'} | BM25 ${row.bm25_rank || '-'} | fused ${row.rank}`}</title>
                  </g>
                )
              })}
              <text x="120" y="276" textAnchor="middle" className="fusion-lane-miss">missing ranks sit low</text>
            </svg>
          </div>
          <div className="fusion-bump-legend"><span><i />Vector</span><span><b />BM25</span><span><em />Fused</span></div>
        </article>

        <article className="viz-card">
          <h4>Rank Fusion Table</h4>
          <div className="fusion-table">
            <div className="fusion-row fusion-head"><span>Rank</span><span>Source</span><span>Vector</span><span>BM25</span><span>Fusion</span></div>
            {rankFusionTable.slice(0, 8).map((row) => (
              <div key={row.chunk_id} className="fusion-row" style={{ borderColor: hoverRank === row.rank ? 'var(--accent)' : 'var(--line)' }}
                onMouseEnter={() => setHoverRank(row.rank)} onMouseLeave={() => setHoverRank(null)}>
                <span>#{row.rank}</span>
                <span><span className={`source-badge compact ${row.source}`}>{sourceLabel(row.source)}</span></span>
                <span>{row.vector_rank ? `#${row.vector_rank}` : '-'}</span>
                <span>{row.bm25_rank ? `#${row.bm25_rank}` : '-'}</span>
                <span>{row.fusion_score?.toFixed?.(3) ?? row.fusion_score}</span>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
