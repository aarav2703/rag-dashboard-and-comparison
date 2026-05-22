import React, { useEffect, useMemo, useRef, useState } from 'react'
import { select, zoom, zoomIdentity } from 'd3'
import AnswerCriticPanel from './AnswerCriticPanel.jsx'

function nodeColor(type) {
  if (type === 'document') return '#e2e8f0'
  if (type === 'entity') return '#f59e0b'
  if (type === 'claim') return '#7c3aed'
  return '#0ea5e9'
}

function edgeColor(type) {
  if (type === 'mentions') return '#f59e0b'
  if (type === 'co-occurs') return '#0e7490'
  if (type === 'supports') return '#10b981'
  return '#475569'
}

function layoutGraph(nodes, edges, width, height) {
  const typed = nodes.map((node, index) => {
    const ring = node.type === 'document' ? 0 : node.type === 'entity' ? 1 : 2
    const countInRing = nodes.filter((item) => (item.type === 'document' ? 0 : item.type === 'entity' ? 1 : 2) === ring).length || 1
    const ringIndex = nodes.slice(0, index + 1).filter((item) => (item.type === 'document' ? 0 : item.type === 'entity' ? 1 : 2) === ring).length - 1
    const radius = ring === 0 ? 0 : ring === 1 ? Math.min(width, height) * 0.24 : Math.min(width, height) * 0.40
    const angle = (ringIndex / countInRing) * Math.PI * 2 - Math.PI / 2
    return { ...node, x: width / 2 + Math.cos(angle) * radius, y: height / 2 + Math.sin(angle) * radius }
  })
  const byId = new Map(typed.map((node) => [node.id, node]))
  return {
    nodes: typed,
    edges: edges
      .map((edge) => ({ ...edge, sourceNode: byId.get(edge.source), targetNode: byId.get(edge.target) }))
      .filter((edge) => edge.sourceNode && edge.targetNode)
  }
}

export default function GraphRagAnalytics({ queryResult, visData }) {
  const [neighborhood, setNeighborhood] = useState('answer')
  const [hoverId, setHoverId] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [visibleTypes, setVisibleTypes] = useState(() => new Set(['document', 'entity', 'claim', 'section']))
  const [transform, setTransform] = useState(zoomIdentity)
  const svgRef = useRef(null)
  const results = queryResult?.results || []
  const graph = neighborhood === 'answer' ? visData?.highlighted_subgraph : visData?.graph
  const nodes = graph?.nodes || []
  const edges = graph?.edges || []
  const pathRows = visData?.query_path || queryResult?.path_explanation || []
  const communities = visData?.communities || []
  const width = 760
  const height = 520
  const filteredNodes = useMemo(() => nodes.filter((node) => visibleTypes.has(node.type || 'section')).slice(0, 110), [nodes, visibleTypes])
  const visibleNodeIds = useMemo(() => new Set(filteredNodes.map((node) => node.id)), [filteredNodes])
  const filteredEdges = useMemo(() => edges.filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)).slice(0, 220), [edges, visibleNodeIds])
  const layout = useMemo(() => layoutGraph(filteredNodes, filteredEdges, width, height), [filteredNodes, filteredEdges])
  const activeId = hoverId || selectedId
  const activeNode = layout.nodes.find((node) => node.id === activeId)
  const connectedIds = useMemo(() => {
    if (!activeId) return new Set()
    const ids = new Set([activeId])
    layout.edges.forEach((edge) => {
      if (edge.source === activeId || edge.source?.id === activeId) ids.add(edge.target?.id || edge.target)
      if (edge.target === activeId || edge.target?.id === activeId) ids.add(edge.source?.id || edge.source)
    })
    return ids
  }, [activeId, layout.edges])

  useEffect(() => {
    if (!svgRef.current) return
    const behavior = zoom().scaleExtent([0.55, 8]).on('zoom', (event) => setTransform(event.transform))
    select(svgRef.current).call(behavior)
    return () => { select(svgRef.current).on('.zoom', null) }
  }, [])

  useEffect(() => { setHoverId(null); setSelectedId(null); setTransform(zoomIdentity) }, [neighborhood])

  function toggleType(type) {
    setVisibleTypes((current) => {
      const next = new Set(current)
      if (next.has(type) && next.size > 1) next.delete(type)
      else next.add(type)
      return next
    })
  }

  if (!nodes.length) {
    return (
      <section className="panel graph-panel">
        <h3>Evidence Knowledge Graph</h3>
        <div className="visual-empty"><strong>No graph evidence yet</strong><span>Run GraphRAG-lite and ask a question to generate entity paths and cited sections.</span></div>
      </section>
    )
  }

  return (
    <section className="panel graph-panel">
      <div className="graph-head">
        <div><h3>Evidence Knowledge Graph</h3><p className="coverage-note">GraphRAG-lite retrieves through entity and section relationships, not just flat similarity.</p></div>
        <div className="graph-filter">
          <button className={neighborhood === 'answer' ? 'active' : ''} onClick={() => setNeighborhood('answer')}>Answer subgraph</button>
          <button className={neighborhood === 'full' ? 'active' : ''} onClick={() => setNeighborhood('full')}>Neighborhood</button>
        </div>
      </div>

      <div className="graph-layout">
        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <AnswerCriticPanel queryResult={queryResult} title="GraphRAG Self-Healing Answer" />
        </article>

        <article className="viz-card graph-canvas-card">
          <div className="graph-canvas-wrap">
            <svg ref={svgRef} className="graph-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
              <rect x="0" y="0" width={width} height={height} className="graph-bg" />
              <g transform={transform.toString()}>
                {communities.slice(0, 5).map((community, index) => (
                  <circle key={community.id} cx={width / 2} cy={height / 2} r={110 + index * 44} className="community-bubble" />
                ))}
                {layout.edges.map((edge, index) => {
                  const isActive = activeId && edge.source !== activeId && edge.target !== activeId
                  const srcX = edge.sourceNode?.x || 0, srcY = edge.sourceNode?.y || 0
                  const tgtX = edge.targetNode?.x || 0, tgtY = edge.targetNode?.y || 0
                  const highlighted = activeNode && (edge.source === activeId || edge.target === activeId)
                  return (
                    <line key={`${edge.source}-${edge.target}-${edge.type}-${index}`}
                      x1={srcX} y1={srcY} x2={tgtX} y2={tgtY}
                      stroke={highlighted ? '#f59e0b' : edgeColor(edge.type)}
                      className={`graph-edge ${isActive ? 'dimmed' : ''} ${highlighted ? 'retrieval-trail' : ''}`}
                      strokeWidth={highlighted ? 2.5 : 1.2} />
                  )
                })}
                {layout.nodes.map((node) => {
                  const isDimmed = activeId && !connectedIds.has(node.id)
                  const isSelected = selectedId === node.id
                  return (
                    <g key={node.id}
                      className={`graph-node ${isDimmed ? 'dimmed' : ''} ${isSelected ? 'selected' : ''}`}
                      onMouseEnter={() => setHoverId(node.id)} onMouseLeave={() => setHoverId(null)}
                      onClick={() => setSelectedId((current) => current === node.id ? null : node.id)}>
                      <circle cx={node.x} cy={node.y}
                        r={node.type === 'document' ? 14 : node.type === 'entity' ? 9 : node.type === 'claim' ? 7 : 7}
                        fill={nodeColor(node.type)} />
                      <text x={node.x + 10} y={node.y + 4}>{node.label}</text>
                    </g>
                  )
                })}
              </g>
            </svg>
            {activeNode && (
              <div className="graph-hover-card">
                <strong>{activeNode.label}</strong>
                <span>{activeNode.type}{activeNode.page_number ? ` | page ${activeNode.page_number}` : ''}</span>
                <p>{activeNode.preview || `${connectedIds.size - 1} connected nodes`}</p>
              </div>
            )}
          </div>
          <div className="zoom-hint">Wheel to zoom. Drag to pan. Hover nodes to inspect; click to pin a neighborhood.</div>
          <div className="graph-legend">
            {['document', 'entity', 'claim', 'section'].map((type) => (
              <button key={type} type="button" className={visibleTypes.has(type) ? 'active' : ''} onClick={() => toggleType(type)}>
                <i className={type} />{type}
              </button>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Query Path Explanation</h4>
          <div className="query-path-list">
            {pathRows.slice(0, 10).map((path, index) => (
              <div key={`${path.entity}-${path.section}-${index}`} className="query-path-row">
                <span>{'\u2192'}</span>
                <div><strong>{path.entity || 'fallback'}</strong><span>{path.related_entity ? `via ${path.related_entity}` : path.edge_type}</span><strong>{path.section || 'section match'}</strong></div>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Cited Graph Evidence</h4>
          <div className="graph-evidence-list">
            {results.map((result) => (
              <div key={result.chunk_id} className="graph-evidence-card">
                <strong>#{result.rank} {result.section_label} | {result.graph_score?.toFixed?.(2) ?? result.graph_score}</strong>
                <div className="graph-entity-tags">
                  {(result.matched_entities || []).slice(0, 6).map((entity) => <span key={entity}>{entity}</span>)}
                </div>
                <p>{result.chunk_text_preview}</p>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
