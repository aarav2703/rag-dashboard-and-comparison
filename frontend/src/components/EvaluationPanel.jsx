import React, { useEffect, useState } from 'react'

const API_BASE = 'http://localhost:5000'

const METHOD_LABELS = {
  naive: 'Naive Vector', bm25: 'BM25', hybrid: 'Hybrid',
  rerank: 'Rerank', graph: 'GraphRAG-lite', vectorless: 'Vectorless Tree',
  agentic: 'Agentic', multihop: 'Multi-hop'
}

const METRIC_LABELS = {
  'ndcg@5': 'NDCG@5', 'mrr': 'MRR', 'recall@5': 'Recall@5', 'precision@5': 'Precision@5'
}

function scoreColor(score, best) {
  if (!best) return 'low'
  if (score >= best * 0.95) return 'top'
  if (score >= best * 0.75) return 'mid'
  return 'low'
}

function RadarChart({ metrics, methods }) {
  if (!methods.length || !metrics) return null
  const metricKeys = Object.keys(METRIC_LABELS)
  const center = 160
  const radius = 130
  const n = metricKeys.length
  const angleStep = (Math.PI * 2) / n

  const getPoint = (value, i) => {
    const angle = angleStep * i - Math.PI / 2
    return { x: center + Math.cos(angle) * radius * value, y: center + Math.sin(angle) * radius * value }
  }

  const colors = ['#f59e0b', '#0ea5e9', '#10b981', '#ef4444', '#7c3aed', '#f97316', '#ec4899', '#06b6d4']

  return (
    <div className="radar-chart-wrap">
      <svg className="radar-svg" viewBox="0 0 320 320" preserveAspectRatio="xMidYMid meet">
        {[0.25, 0.5, 0.75].map((level, li) => (
          <polygon
            key={`grid-${li}`}
            points={metricKeys.map((_, i) => {
              const p = getPoint(level, i)
              return `${p.x},${p.y}`
            }).join(' ')}
            fill="none"
            stroke="rgba(148,163,184,0.15)"
            strokeWidth="1"
          />
        ))}
        {metricKeys.map((_, i) => (
          <line
            key={`axis-${i}`}
            x1={center} y1={center}
            x2={getPoint(1, i).x} y2={getPoint(1, i).y}
            stroke="rgba(148,163,184,0.15)"
            strokeWidth="1"
          />
        ))}
        {methods.map((method, mi) => {
          const vals = metricKeys.map(k => metrics[k]?.[method.id] || 0)
          return (
            <polygon
              key={method.id}
              points={vals.map((v, i) => {
                const p = getPoint(v, i)
                return `${p.x},${p.y}`
              }).join(' ')}
              fill={colors[mi] + '18'}
              stroke={colors[mi]}
              strokeWidth="2"
            />
          )
        })}
        {metricKeys.map((key, i) => {
          const p = getPoint(1.12, i)
          return (
            <text key={`label-${key}`} x={p.x} y={p.y} textAnchor="middle" fill="var(--muted)" fontSize="10" fontWeight="700">
              {METRIC_LABELS[key]}
            </text>
          )
        })}
      </svg>
    </div>
  )
}

export default function EvaluationPanel({ queryText }) {
  const [evalData, setEvalData] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!queryText) return
    setLoading(true)
    fetch(`${API_BASE}/api/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: queryText })
    })
      .then(res => res.json())
      .then(data => {
        setEvalData(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [queryText])

  if (!evalData && !loading) return null
  if (loading) return (
    <section className="panel eval-panel">
      <h3>Evaluation Metrics</h3>
      <div className="comparison-empty">Computing evaluation metrics across all methods...</div>
    </section>
  )
  if (evalData?.error) return (
    <section className="panel eval-panel">
      <h3>Evaluation Metrics</h3>
      <div className="comparison-empty">{evalData.error}</div>
    </section>
  )

  const { comparison } = evalData
  if (!comparison?.table) return null

  const metrics = comparison.table
  const methods = (comparison.methods || []).map(m => ({ id: m, label: METHOD_LABELS[m] || m }))
  const ranking = comparison.ranking || {}
  const bestMethod = ranking['ndcg@5']?.[0]?.method
  const sortedByNdcg = methods.slice().sort((a, b) => (metrics['ndcg@5']?.[b.id] || 0) - (metrics['ndcg@5']?.[a.id] || 0))

  return (
    <section className="panel eval-panel">
      <h3>Evaluation Metrics</h3>

      {bestMethod && (
        <div className="eval-highlight-box">
          <h4>Top Performer: {METHOD_LABELS[bestMethod] || bestMethod}</h4>
          <p>NDCG@5: {metrics['ndcg@5']?.[bestMethod]?.toFixed(3)} &middot; MRR: {metrics['mrr']?.[bestMethod]?.toFixed(3)} &middot; Recall@5: {metrics['recall@5']?.[bestMethod]?.toFixed(3)}</p>
        </div>
      )}

      <article className="viz-card">
        <h4>Performance Radar</h4>
        <RadarChart metrics={metrics} methods={methods} />
      </article>

      <article className="viz-card">
        <h4>Metric Leaderboard</h4>
        <div className="eval-table">
          <div className="eval-table-header">
            <span>Method</span>
            <span>NDCG@5</span>
            <span>MRR</span>
            <span>Recall@5</span>
            <span>Prec@5</span>
            <span>NDCG@10</span>
          </div>
          {sortedByNdcg.map((method, i) => {
            const best = metrics['ndcg@5']?.[sortedByNdcg[0]?.id] || 1
            const s = metrics['ndcg@5']?.[method.id] || 0
            return (
              <div key={method.id} className={`eval-table-row ${i === 0 ? 'best' : ''} ${i >= sortedByNdcg.length - 2 ? 'worst' : ''}`}>
                <span>
                  {i === 0 && <span className="eval-winner-badge">best</span>}
                  {i === 0 ? ' ' : ''}{method.label}
                </span>
                <span className={`eval-score ${scoreColor(s, best)}`}>{s.toFixed(3)}</span>
                <span className="eval-score">{metrics['mrr']?.[method.id]?.toFixed(3) || '-'}</span>
                <span className="eval-score">{metrics['recall@5']?.[method.id]?.toFixed(3) || '-'}</span>
                <span className="eval-score">{metrics['precision@5']?.[method.id]?.toFixed(3) || '-'}</span>
                <span className="eval-score">{metrics['ndcg@10']?.[method.id]?.toFixed(3) || '-'}</span>
              </div>
            )
          })}
        </div>
      </article>
    </section>
  )
}
