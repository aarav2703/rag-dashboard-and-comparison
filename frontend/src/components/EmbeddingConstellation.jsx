import React, { useEffect, useMemo, useRef, useState } from 'react'
import { select, zoom, zoomIdentity } from 'd3'

function hashString(input) {
  let h = 0
  for (let i = 0; i < input.length; i += 1) h = (h * 31 + input.charCodeAt(i)) >>> 0
  return h
}

function deterministicCoord(id, scale) {
  const h = hashString(id || 'unknown')
  return ((((h % 10000) / 10000) * 2) - 1) * scale
}

export default function EmbeddingConstellation({ chunks, queryResult, visData }) {
  const [hoverId, setHoverId] = useState(null)
  const [transform, setTransform] = useState(zoomIdentity)
  const svgRef = useRef(null)
  const width = 920
  const height = 620

  useEffect(() => {
    if (!svgRef.current) return
    const behavior = zoom().scaleExtent([0.5, 8]).on('zoom', (event) => setTransform(event.transform))
    const svg = select(svgRef.current)
    svg.call(behavior)
    return () => { svg.on('.zoom', null) }
  }, [])

  const points = useMemo(() => {
    const map = new Map()
    ;(visData?.points || []).forEach((p) => {
      map.set(p.chunk_id, { id: p.chunk_id, page: p.page_number, x: p.x, y: p.y, preview: p.preview || '', similarity: p.similarity_score || 0, isRetrieved: Boolean(p.is_retrieved) })
    })
    ;(chunks || []).forEach((c) => {
      if (!map.has(c.chunk_id)) {
        map.set(c.chunk_id, { id: c.chunk_id, page: c.page_number, x: deterministicCoord(`${c.chunk_id}:x`, 1), y: deterministicCoord(`${c.chunk_id}:y`, 1), preview: c.preview || (c.chunk_text || '').slice(0, 220), similarity: 0, isRetrieved: false })
      }
    })
    return Array.from(map.values())
  }, [chunks, visData])

  const scaledPoints = useMemo(() => {
    if (!points.length) return []
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y)
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
    return points.map((p) => {
      const nx = maxX === minX ? 0.5 : (p.x - minX) / (maxX - minX)
      const ny = maxY === minY ? 0.5 : (p.y - minY) / (maxY - minY)
      return { ...p, sx: 30 + nx * (width - 60), sy: 30 + ny * (height - 60) }
    })
  }, [points])

  const queryPoint = useMemo(() => {
    if (!queryResult?.query) return null
    const qp = visData?.query_point
    if (!qp || typeof qp.x !== 'number' || typeof qp.y !== 'number') return { x: width / 2, y: height / 2, query: queryResult.query }
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y)
    const minX = xs.length ? Math.min(...xs) : -1, maxX = xs.length ? Math.max(...xs) : 1
    const minY = ys.length ? Math.min(...ys) : -1, maxY = ys.length ? Math.max(...ys) : 1
    const nx = maxX === minX ? 0.5 : (qp.x - minX) / (maxX - minX)
    const ny = maxY === minY ? 0.5 : (qp.y - minY) / (maxY - minY)
    return { x: 30 + nx * (width - 60), y: 30 + ny * (height - 60), query: qp.query || queryResult.query }
  }, [points, queryResult, visData])

  const results = queryResult?.results || []
  const maxSimilarity = results.length ? Math.max(...results.map((r) => r.similarity_score || 0)) : 1
  const hoverPoint = scaledPoints.find((p) => p.id === hoverId)
  const baseRadius = scaledPoints.length > 1800 ? 1.7 : scaledPoints.length > 900 ? 2.3 : 4
  const retrievedRadius = scaledPoints.length > 1800 ? 4.5 : scaledPoints.length > 900 ? 5 : 6

  if (!scaledPoints.length) {
    return (
      <section className="panel constellation-panel">
        <h3>Interactive Embedding Space</h3>
        <div className="visual-empty"><strong>No embedding projection yet</strong><span>Run the Naive Vector RAG pipeline to generate chunk coordinates.</span></div>
      </section>
    )
  }

  return (
    <section className="panel constellation-panel">
      <div className="constellation-head">
        <div><h3>Interactive Embedding Space</h3><div className="zoom-hint">Wheel to zoom. Drag to pan. Hover nodes to inspect chunks.</div></div>
        <div className="constellation-stats">
          <span>{scaledPoints.length} points</span>
          <span>{chunks?.length || 0} chunks</span>
          <span>{results.length} retrieved</span>
        </div>
      </div>
      <div className="constellation">
        <div className="svg-wrap">
          <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" className="constellation-svg">
            <rect x={0} y={0} width={width} height={height} rx={14} className="bg" />
            <g transform={transform.toString()}>
              {queryPoint && results.map((r, idx) => {
                const target = scaledPoints.find((p) => p.id === r.chunk_id)
                if (!target) return null
                const dist = Math.sqrt((queryPoint.x - target.sx) ** 2 + (queryPoint.y - target.sy) ** 2)
                return (
                  <line key={`line-${idx}`} x1={queryPoint.x} y1={queryPoint.y} x2={target.sx} y2={target.sy}
                    className="radial-line retrieval-trail"
                    strokeDasharray={`${Math.max(2, dist * 0.18)} ${Math.max(4, dist * 0.12)}`} />
                )
              })}
              {scaledPoints.map((p) => (
                <circle key={p.id} cx={p.sx} cy={p.sy}
                  r={p.isRetrieved ? retrievedRadius : baseRadius}
                  className={p.isRetrieved ? 'node retrieved particle-dot' : 'node'}
                  onMouseEnter={() => setHoverId(p.id)} onMouseLeave={() => setHoverId(null)} />
              ))}
              {queryPoint && (
                <g>
                  <circle cx={queryPoint.x} cy={queryPoint.y} r={12} className="query-node" />
                  <text x={queryPoint.x + 16} y={queryPoint.y - 8} className="query-label">{queryPoint.query.slice(0, 30)}</text>
                </g>
              )}
            </g>
          </svg>
          {hoverPoint && (
            <div className="hover-card">
              <div className="hover-title">Chunk {hoverPoint.id.slice(0, 8)}...</div>
              <div className="hover-meta">Page {hoverPoint.page} | score {hoverPoint.similarity.toFixed(3)}</div>
              <div className="hover-preview">{hoverPoint.preview}</div>
            </div>
          )}
        </div>
        <aside className="side">
          <h4>Query</h4>
          <div className="query-text">{queryResult?.query || 'No query yet'}</div>
          <h4>Cosine Similarity Bars</h4>
          <div className="results">
            {results.map((r) => {
              const widthPct = maxSimilarity > 0 ? ((r.similarity_score || 0) / maxSimilarity) * 100 : 0
              return (
                <div key={r.chunk_id} className={`result-item ${hoverId === r.chunk_id ? 'hover' : ''}`}
                  onMouseEnter={() => setHoverId(r.chunk_id)} onMouseLeave={() => setHoverId(null)}>
                  <div className="meta">#{r.rank} page {r.page_number} | {r.similarity_score.toFixed(3)}</div>
                  <div className="bar-track"><div className="bar-fill" style={{ width: `${widthPct}%`, transition: 'width 420ms ease' }} /></div>
                  <div className="preview">{r.chunk_text_preview}</div>
                </div>
              )
            })}
          </div>
        </aside>
      </div>
    </section>
  )
}
