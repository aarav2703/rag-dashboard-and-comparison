import React, { useMemo, useState } from 'react'
import AnswerCriticPanel from './AnswerCriticPanel.jsx'

function yForRank(rank, maxRank, height) {
  if (maxRank <= 1) return height / 2
  return 24 + ((rank - 1) / (maxRank - 1)) * (height - 48)
}

function movementClass(movement) {
  if (movement > 0) return 'promoted'
  if (movement < 0) return 'demoted'
  return 'unchanged'
}

export default function RerankAnalytics({ queryResult, visData }) {
  const [hoverId, setHoverId] = useState(null)
  const [animPhase, setAnimPhase] = useState(0)
  const rows = visData?.slopegraph || queryResult?.before_after_table || []
  const results = queryResult?.results || []
  const histogram = visData?.reranker_score_histogram || queryResult?.reranker_score_histogram || []
  const promoted = visData?.promoted_chunks || queryResult?.promoted_chunks || []
  const demoted = visData?.demoted_chunks || queryResult?.demoted_chunks || []
  const summary = visData?.summary || {}

  const slopeRows = useMemo(() => rows.slice(0, 14), [rows])
  const maxRank = Math.max(...slopeRows.flatMap((row) => [row.before_rank || 1, row.after_rank || 1]), 5)
  const height = Math.max(320, maxRank * 22)
  const maxBucket = Math.max(...histogram.map((bucket) => bucket.count || 0), 1)
  const hoverRow = rows.find((row) => row.chunk_id === hoverId)

  React.useEffect(() => {
    const interval = setInterval(() => setAnimPhase((p) => p + 1), 2000)
    return () => clearInterval(interval)
  }, [])

  if (!rows.length) {
    return (
      <section className="panel rerank-panel">
        <h3>Rerank Movement</h3>
        <div className="visual-empty"><strong>No rerank data yet</strong><span>Run Rerank RAG and ask a question to see candidate rank movement.</span></div>
      </section>
    )
  }

  return (
    <section className="panel rerank-panel">
      <h3>Rerank Movement</h3>

      <div className="rerank-summary-grid">
        <div><span>Candidates</span><strong>{summary.candidate_count || rows.length}</strong></div>
        <div><span>Promoted</span><strong style={{ color: 'var(--tertiary)' }}>{summary.promoted_count ?? promoted.length}</strong></div>
        <div><span>Demoted</span><strong style={{ color: 'var(--danger)' }}>{summary.demoted_count ?? demoted.length}</strong></div>
        <div><span>Final evidence</span><strong>{results.length}</strong></div>
      </div>

      <div className="rerank-grid">
        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <AnswerCriticPanel queryResult={queryResult} title="Rerank Self-Healing Answer" />
        </article>

        <article className="viz-card rerank-slope-card">
          <h4>Rank Movement Slopegraph</h4>
          <div className="coverage-note">Left is initial FAISS candidate rank. Right is final rank after reranking.</div>
          <svg className="rerank-slope-svg" viewBox={`0 0 640 ${height}`} preserveAspectRatio="xMidYMid meet">
            <text x="70" y="16" className="rerank-axis-label">Before rerank</text>
            <text x="470" y="16" className="rerank-axis-label">After rerank</text>
            <line x1="130" y1="24" x2="130" y2={height - 24} className="rerank-axis" />
            <line x1="510" y1="24" x2="510" y2={height - 24} className="rerank-axis" />
            {slopeRows.map((row, idx) => {
              const y1 = yForRank(row.before_rank, maxRank, height)
              const y2 = yForRank(row.after_rank, maxRank, height)
              const cls = movementClass(row.movement)
              return (
                <g key={row.chunk_id}
                  className={hoverId && hoverId !== row.chunk_id ? 'rerank-dimmed' : ''}
                  style={{
                    opacity: animPhase % slopeRows.length >= idx ? 1 : 0.25,
                    transition: 'opacity 400ms ease'
                  }}
                  onMouseEnter={() => setHoverId(row.chunk_id)} onMouseLeave={() => setHoverId(null)}>
                  <line x1="130" y1={y1} x2="510" y2={y2} className={`rerank-slope-line ${cls}`} />
                  <circle cx="130" cy={y1} r="4" className={`rerank-dot ${cls}`} />
                  <circle cx="510" cy={y2} r="4" className={`rerank-dot ${cls}`} />
                  <text x="18" y={y1 + 4} className="rerank-rank-label">#{row.before_rank}</text>
                  <text x="528" y={y2 + 4} className="rerank-rank-label">#{row.after_rank}</text>
                  <text x="150" y={(y1 + y2) / 2 - 4} className="rerank-chunk-label">P{row.page_number} {row.movement > 0 ? `\u2191${row.movement}` : row.movement < 0 ? `\u2193${Math.abs(row.movement)}` : '\u2194'}</text>
                </g>
              )
            })}
          </svg>
          {hoverRow && (
            <div className="rerank-hover-card">
              <strong>P{hoverRow.page_number} moved {hoverRow.movement > 0 ? `up ${hoverRow.movement}` : hoverRow.movement < 0 ? `down ${Math.abs(hoverRow.movement)}` : 'nowhere'}</strong>
              <span>before #{hoverRow.before_rank} to after #{hoverRow.after_rank} | score {hoverRow.reranker_score?.toFixed?.(3) ?? hoverRow.reranker_score}</span>
              <p>{hoverRow.preview || hoverRow.chunk_text_preview || 'No preview available.'}</p>
            </div>
          )}
        </article>

        <article className="viz-card">
          <h4>Reranker Score Histogram</h4>
          <div className="rerank-histogram">
            {histogram.map((bucket) => (
              <div key={bucket.bucket} className="rerank-histogram-row">
                <span>{bucket.start.toFixed?.(2) ?? bucket.start}-{bucket.end.toFixed?.(2) ?? bucket.end}</span>
                <div className="rerank-histogram-track">
                  <div style={{ width: `${Math.max(4, (bucket.count / maxBucket) * 100)}%`, transition: 'width 600ms ease' }} />
                </div>
                <strong>{bucket.count}</strong>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card rerank-table-card">
          <h4>Before / After Top-K Table</h4>
          <div className="rerank-table">
            <div className="rerank-table-row rerank-table-head"><span>Chunk</span><span>Before</span><span>After</span><span>Move</span><span>Score</span></div>
            {rows.slice(0, 10).map((row) => (
              <div key={row.chunk_id} className={`rerank-table-row ${hoverId === row.chunk_id ? 'active' : ''}`}
                onMouseEnter={() => setHoverId(row.chunk_id)} onMouseLeave={() => setHoverId(null)}>
                <span>P{row.page_number}</span>
                <span>#{row.before_rank}</span>
                <span>#{row.after_rank}</span>
                <span className={movementClass(row.movement)}>{row.movement > 0 ? `+${row.movement}` : row.movement}</span>
                <span>{row.reranker_score?.toFixed?.(3) ?? row.reranker_score}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Promoted / Demoted Chunks</h4>
          <div className="rerank-movement-cards">
            {[...promoted.slice(0, 3), ...demoted.slice(0, 3)].map((row) => (
              <div key={`${row.chunk_id}-${row.movement}`} className={`rerank-movement-card ${movementClass(row.movement)}`}>
                <strong>P{row.page_number} {row.movement > 0 ? `promoted +${row.movement}` : `demoted ${row.movement}`}</strong>
                <p>{row.preview}</p>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
