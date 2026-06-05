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

function gradeCounts(grades) {
  return grades.reduce((acc, row) => {
    const key = row.grade || 'unknown'
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, {})
}

export default function CragAnalytics({ queryResult, visData }) {
  const [hoverId, setHoverId] = useState(null)
  const [animPhase, setAnimPhase] = useState(0)
  const rows = visData?.slopegraph || queryResult?.before_after_table || []
  const results = queryResult?.results || []
  const histogram = visData?.reranker_score_histogram || queryResult?.reranker_score_histogram || []
  const promoted = visData?.promoted_chunks || queryResult?.promoted_chunks || []
  const demoted = visData?.demoted_chunks || queryResult?.demoted_chunks || []
  const summary = visData?.summary || {}
  const cragSummary = queryResult?.crag_summary || visData?.crag_flow || {}
  const grades = queryResult?.evidence_grades || visData?.evidence_grades || []
  const webResults = queryResult?.web_results || visData?.web_results || []
  const correctionAttempts = queryResult?.correction_attempts || visData?.correction_attempts || []

  const slopeRows = useMemo(() => rows.slice(0, 14), [rows])
  const maxRank = Math.max(...slopeRows.flatMap((row) => [row.before_rank || 1, row.after_rank || 1]), 5)
  const height = Math.max(320, maxRank * 22)
  const maxBucket = Math.max(...histogram.map((bucket) => bucket.count || 0), 1)
  const hoverRow = rows.find((row) => row.chunk_id === hoverId)
  const counts = gradeCounts(grades)
  const totalGrades = Math.max(1, grades.length)
  const activeBranch = cragSummary.branch || queryResult?.retrieval_verdict || 'pending'
  const fallbackSource = cragSummary.fallback_source || queryResult?.fallback_source || 'none'
  const criticVerdict = queryResult?.critic?.verdict || 'pending'

  React.useEffect(() => {
    const interval = setInterval(() => setAnimPhase((p) => p + 1), 2000)
    return () => clearInterval(interval)
  }, [])

  if (!rows.length) {
    return (
      <section className="panel rerank-panel">
        <h3>Corrective RAG Flow</h3>
        <div className="visual-empty"><strong>No corrective trace yet</strong><span>Run CRAG and ask a question to see reranking, evidence grading, correction, and grounded answer checking.</span></div>
      </section>
    )
  }

  return (
    <section className="panel rerank-panel">
      <h3>Corrective RAG with Reranking</h3>

      <div className="rerank-summary-grid">
        <div><span>Candidates</span><strong>{summary.candidate_count || rows.length}</strong></div>
        <div><span>Promoted</span><strong style={{ color: 'var(--tertiary)' }}>{summary.promoted_count ?? promoted.length}</strong></div>
        <div><span>Branch</span><strong style={{ color: 'var(--danger)' }}>{cragSummary.branch || '-'}</strong></div>
        <div><span>Action</span><strong>{cragSummary.action || '-'}</strong></div>
        <div><span>Grader</span><strong>{cragSummary.grader_source || '-'}</strong></div>
        <div><span>Fallback</span><strong>{cragSummary.fallback_source || queryResult?.fallback_source || '-'}</strong></div>
      </div>

      <div className="rerank-grid">
        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <AnswerCriticPanel queryResult={queryResult} title="CRAG Grounded Answer" />
        </article>

        <article className="viz-card crag-tree-card" style={{ gridColumn: '1 / -1' }}>
          <h4>Corrective Decision Tree</h4>
          <div className="crag-decision-tree">
            {['Query', 'Retrieve', 'Rerank', 'Grade'].map((label) => (
              <div key={label} className="crag-tree-node active"><span>{label}</span></div>
            ))}
            <div className={`crag-tree-branch correct ${activeBranch === 'correct' ? 'active' : ''}`}>
              <strong>Correct</strong><span>Answer directly</span>
            </div>
            <div className={`crag-tree-branch ambiguous ${activeBranch === 'ambiguous' ? 'active' : ''}`}>
              <strong>Ambiguous</strong><span>Rewrite locally</span>
            </div>
            <div className={`crag-tree-branch incorrect ${activeBranch === 'incorrect' ? 'active' : ''}`}>
              <strong>Incorrect</strong><span>{fallbackSource === 'web' ? 'Tavily fallback' : 'Local fallback'}</span>
            </div>
            <div className={`crag-tree-node gate ${criticVerdict}`}>
              <span>Groundedness Gate</span><strong>{criticVerdict}</strong>
            </div>
          </div>
          <div className="grade-distribution-strip">
            <span className="correct" style={{ width: `${((counts.correct || 0) / totalGrades) * 100}%` }} title={`${counts.correct || 0} correct`} />
            <span className="ambiguous" style={{ width: `${((counts.ambiguous || 0) / totalGrades) * 100}%` }} title={`${counts.ambiguous || 0} ambiguous`} />
            <span className="incorrect" style={{ width: `${((counts.incorrect || 0) / totalGrades) * 100}%` }} title={`${counts.incorrect || 0} incorrect`} />
          </div>
          <div className="grade-distribution-legend">
            <span>Correct {counts.correct || 0}</span>
            <span>Ambiguous {counts.ambiguous || 0}</span>
            <span>Incorrect {counts.incorrect || 0}</span>
          </div>
        </article>

        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <h4>Evaluator / Grader Branch</h4>
          <div className="rerank-summary-grid">
            <div><span>Correct</span><strong>{cragSummary.grade_counts?.correct || 0}</strong></div>
            <div><span>Ambiguous</span><strong>{cragSummary.grade_counts?.ambiguous || 0}</strong></div>
            <div><span>Incorrect</span><strong>{cragSummary.grade_counts?.incorrect || 0}</strong></div>
            <div><span>Fallback</span><strong>{cragSummary.fallback_count || 0}</strong></div>
            <div><span>Grade Conf.</span><strong>{cragSummary.average_grade_confidence?.toFixed?.(2) ?? '-'}</strong></div>
            <div><span>Verdict</span><strong>{cragSummary.retrieval_verdict || queryResult?.retrieval_verdict || '-'}</strong></div>
          </div>
          {(cragSummary.missing_evidence_summary || queryResult?.missing_evidence_summary) && (
            <div className="answer-critic-retry">
              <span>Missing evidence</span>
              <p>{cragSummary.missing_evidence_summary || queryResult.missing_evidence_summary}</p>
            </div>
          )}
          {queryResult?.correction_query && queryResult.correction_query !== queryResult.query && (
            <div className="answer-critic-retry">
              <span>Correction query</span>
              <p>{queryResult.correction_query}</p>
            </div>
          )}
        </article>

        <article className="viz-card rerank-slope-card">
          <h4>Rerank Details: Rank Movement Slopegraph</h4>
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
          <h4>Correction Attempts</h4>
          <div className="agent-timeline">
            {correctionAttempts.length ? correctionAttempts.map((item, index) => (
              <div key={`${item.action}-${index}`} className="agent-timeline-row complete">
                <span>{index + 1}</span>
                <div style={{ flex: 1 }}>
                  <strong style={{ fontSize: 11 }}>{item.action}</strong>
                  <p style={{ margin: '4px 0 0', color: 'var(--muted)', fontSize: 11 }}>{item.query}</p>
                </div>
                <em>{item.source}</em>
              </div>
            )) : <div className="comparison-empty">No correction attempt was needed.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Web Fallback Evidence</h4>
          <div className="agent-rejected-pile">
            {webResults.length ? webResults.map((item) => (
              <div key={item.chunk_id}>
                <strong>{item.title || item.url}</strong>
                <span>{item.provider || 'tavily'}</span>
                <p>{item.snippet || item.chunk_text_preview}</p>
              </div>
            )) : <div className="comparison-empty">No Tavily web fallback evidence used for this query.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Evidence Grades</h4>
          <div className="rerank-movement-cards">
            {(grades.length ? grades : [...promoted.slice(0, 3), ...demoted.slice(0, 3)]).slice(0, 6).map((row) => (
              <div key={`${row.chunk_id}-${row.grade || row.movement}`} className={`rerank-movement-card ${row.grade === 'correct' ? 'promoted' : row.grade === 'incorrect' ? 'demoted' : movementClass(row.movement || 0)}`}>
                <strong>P{row.page_number} {row.grade || (row.movement > 0 ? `promoted +${row.movement}` : `demoted ${row.movement}`)}</strong>
                <p>{row.grade_reason || row.reason || row.preview}</p>
                {typeof row.confidence === 'number' && <span>confidence {row.confidence.toFixed(2)}</span>}
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
