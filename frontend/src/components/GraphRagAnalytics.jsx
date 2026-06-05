import React, { useEffect, useMemo, useRef, useState } from 'react'
import { select, zoom, zoomIdentity } from 'd3'

const WIDTH = 980
const HEIGHT = 640
const NODE_LIMIT = 140
const EDGE_LIMIT = 280
const NODE_TYPES = ['document', 'section', 'entity', 'claim']

const SCORE_KEYS = [
  ['title_entity_match', 'Title'],
  ['direct_entity', 'Entity'],
  ['one_hop_relationship', '1-hop'],
  ['two_hop_relationship', '2-hop'],
  ['vector_seed', 'Vector'],
  ['bm25_seed', 'BM25']
]

function cleanType(type) {
  return NODE_TYPES.includes(type) ? type : 'section'
}

function nodeColor(type) {
  if (type === 'document') return '#e2e8f0'
  if (type === 'entity') return '#f59e0b'
  if (type === 'claim') return '#a78bfa'
  return '#38bdf8'
}

function edgeColor(type = '') {
  const normalized = String(type).toLowerCase()
  if (normalized.includes('mention')) return '#f59e0b'
  if (normalized.includes('support')) return '#10b981'
  if (normalized.includes('co') || normalized.includes('related')) return '#38bdf8'
  if (normalized.includes('path')) return '#a78bfa'
  return '#64748b'
}

function nodeRadius(type, degree = 0) {
  const base = type === 'document' ? 18 : type === 'entity' ? 11 : type === 'claim' ? 8 : 9
  return Math.min(base + Math.sqrt(degree) * 1.2, base + 7)
}

function displayLabel(value = '', max = 34) {
  const label = String(value || '').trim() || 'Untitled'
  return label.length > max ? `${label.slice(0, max - 1)}...` : label
}

function edgeSource(edge) {
  return edge.source?.id || edge.source
}

function edgeTarget(edge) {
  return edge.target?.id || edge.target
}

function scoreEntries(breakdown = {}) {
  return SCORE_KEYS
    .map(([key, label]) => ({ key, label, value: Number(breakdown[key] || 0) }))
    .filter((row) => row.value > 0)
}

function polarPoint(cx, cy, radius, angle) {
  return {
    x: cx + Math.cos(angle) * radius,
    y: cy + Math.sin(angle) * radius
  }
}

function tierLayout(nodes, edges) {
  const degree = new Map()
  edges.forEach((edge) => {
    const source = edgeSource(edge)
    const target = edgeTarget(edge)
    degree.set(source, (degree.get(source) || 0) + 1)
    degree.set(target, (degree.get(target) || 0) + 1)
  })

  const typed = NODE_TYPES.reduce((acc, type) => ({ ...acc, [type]: [] }), {})
  nodes.forEach((node) => typed[cleanType(node.type)].push(node))

  const centerX = WIDTH / 2
  const centerY = HEIGHT / 2
  const radii = { document: 0, section: 170, entity: 260, claim: 338 }
  const angleOffset = { document: -Math.PI / 2, section: -Math.PI * 0.74, entity: -Math.PI / 2, claim: -Math.PI * 0.36 }
  const spread = { document: Math.PI * 2, section: Math.PI * 1.22, entity: Math.PI * 1.78, claim: Math.PI * 1.1 }

  const positioned = []
  NODE_TYPES.forEach((type) => {
    const group = typed[type]
      .map((node) => ({ ...node, type: cleanType(node.type), degree: degree.get(node.id) || 0 }))
      .sort((a, b) => (b.degree - a.degree) || String(a.label).localeCompare(String(b.label)))

    group.forEach((node, index) => {
      if (type === 'document') {
        const row = Math.floor(index / 3)
        const col = index % 3
        positioned.push({
          ...node,
          x: centerX + (col - 1) * 70,
          y: centerY + (row - Math.max(0, Math.ceil(group.length / 3) - 1) / 2) * 54
        })
        return
      }

      const count = Math.max(1, group.length)
      const angle = count === 1
        ? angleOffset[type] + spread[type] / 2
        : angleOffset[type] + (index / (count - 1)) * spread[type]
      const jitter = (index % 2 ? 1 : -1) * Math.min(20, index * 1.7)
      const point = polarPoint(centerX, centerY, radii[type] + jitter, angle)
      positioned.push({ ...node, ...point })
    })
  })

  const byId = new Map(positioned.map((node) => [node.id, node]))
  return {
    nodes: positioned,
    edges: edges
      .map((edge) => ({ ...edge, sourceNode: byId.get(edgeSource(edge)), targetNode: byId.get(edgeTarget(edge)) }))
      .filter((edge) => edge.sourceNode && edge.targetNode)
  }
}

function useGraphZoom(resetKey) {
  const svgRef = useRef(null)
  const zoomRef = useRef(null)
  const [transform, setTransform] = useState(zoomIdentity)

  useEffect(() => {
    if (!svgRef.current) return undefined
    const behavior = zoom().scaleExtent([0.45, 7]).on('zoom', (event) => setTransform(event.transform))
    zoomRef.current = behavior
    select(svgRef.current).call(behavior)
    return () => select(svgRef.current).on('.zoom', null)
  }, [])

  useEffect(() => {
    if (svgRef.current && zoomRef.current) {
      select(svgRef.current).call(zoomRef.current.transform, zoomIdentity)
    } else {
      setTransform(zoomIdentity)
    }
  }, [resetKey])

  function applyTransform(nextTransform) {
    if (svgRef.current && zoomRef.current) select(svgRef.current).call(zoomRef.current.transform, nextTransform)
    else setTransform(nextTransform)
  }

  function zoomTo(scale) {
    const nextScale = Math.max(0.45, Math.min(7, scale))
    applyTransform(zoomIdentity.translate(transform.x, transform.y).scale(nextScale))
  }

  return { svgRef, transform, applyTransform, zoomTo }
}

export default function GraphRagAnalytics({ queryResult, visData }) {
  const [neighborhood, setNeighborhood] = useState('answer')
  const [hoverId, setHoverId] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [visibleTypes, setVisibleTypes] = useState(() => new Set(NODE_TYPES))
  const { svgRef, transform, applyTransform, zoomTo } = useGraphZoom(neighborhood)

  const results = queryResult?.results || []
  const graph = neighborhood === 'answer' ? visData?.highlighted_subgraph : visData?.graph
  const rawNodes = graph?.nodes || []
  const rawEdges = graph?.edges || []
  const pathRows = visData?.query_path || queryResult?.path_explanation || []
  const communities = visData?.communities || []
  const communitySummaries = visData?.community_summaries || queryResult?.community_summaries || []
  const relationships = visData?.relationships || queryResult?.relationships || []
  const relationshipPaths = visData?.relationship_paths || queryResult?.relationship_paths || []
  const communityHits = visData?.community_hits || queryResult?.community_hits || []
  const entityMatches = queryResult?.entity_matches || []

  const filteredNodes = useMemo(() => {
    return rawNodes
      .map((node) => ({ ...node, type: cleanType(node.type) }))
      .filter((node) => visibleTypes.has(node.type))
      .slice(0, NODE_LIMIT)
  }, [rawNodes, visibleTypes])

  const visibleNodeIds = useMemo(() => new Set(filteredNodes.map((node) => node.id)), [filteredNodes])

  const filteredEdges = useMemo(() => {
    return rawEdges
      .filter((edge) => visibleNodeIds.has(edgeSource(edge)) && visibleNodeIds.has(edgeTarget(edge)))
      .slice(0, EDGE_LIMIT)
  }, [rawEdges, visibleNodeIds])

  const layout = useMemo(() => tierLayout(filteredNodes, filteredEdges), [filteredNodes, filteredEdges])
  const activeId = selectedId || hoverId
  const activeNode = layout.nodes.find((node) => node.id === activeId)

  const connectedIds = useMemo(() => {
    if (!activeId) return new Set()
    const ids = new Set([activeId])
    layout.edges.forEach((edge) => {
      const source = edgeSource(edge)
      const target = edgeTarget(edge)
      if (source === activeId) ids.add(target)
      if (target === activeId) ids.add(source)
    })
    return ids
  }, [activeId, layout.edges])

  function toggleType(type) {
    setVisibleTypes((current) => {
      const next = new Set(current)
      if (next.has(type) && next.size > 1) next.delete(type)
      else next.add(type)
      return next
    })
  }

  function clearFocus() {
    setHoverId(null)
    setSelectedId(null)
  }

  if (!rawNodes.length) {
    return (
      <section className="panel graph-panel">
        <h3>Evidence Knowledge Graph</h3>
        <div className="visual-empty"><strong>No graph evidence yet</strong><span>Run GraphRAG and ask a question to generate entity paths and cited sections.</span></div>
      </section>
    )
  }

  return (
    <section className="panel graph-panel">
      <div className="graph-head">
        <div>
          <h3>Evidence Knowledge Graph</h3>
          <p className="coverage-note">GraphRAG retrieves through entities, sections, communities, and relationship paths.</p>
        </div>
        <div className="graph-filter">
          <button type="button" className={neighborhood === 'answer' ? 'active' : ''} onClick={() => setNeighborhood('answer')}>Answer subgraph</button>
          <button type="button" className={neighborhood === 'full' ? 'active' : ''} onClick={() => setNeighborhood('full')}>Neighborhood</button>
        </div>
      </div>

      <div className="graph-layout graph-layout-first">
        <article className="viz-card graph-answer-card">
          <h4>GraphRAG Answer</h4>
          <p className="answer-critic-text">{queryResult?.answer || results[0]?.chunk_text_preview || 'No answer available yet.'}</p>
          <div className="answer-critic-meta">
            <span>Evidence {queryResult?.evidence_count ?? results.length}</span>
            <span>{layout.nodes.length}/{rawNodes.length} nodes shown</span>
            <span>{layout.edges.length}/{rawEdges.length} edges shown</span>
          </div>
        </article>

        <article className="viz-card graph-canvas-card graph-hero-card">
          <div className="graph-card-title">
            <div>
              <h4>Evidence Knowledge Graph Explorer</h4>
              <p>Hover to isolate a neighborhood. Click to pin it; reset clears the focus.</p>
            </div>
            <div className="graph-mini-stats">
              <span>{neighborhood === 'answer' ? 'answer graph' : 'full neighborhood'}</span>
              <span>{transform.k.toFixed(1)}x</span>
            </div>
          </div>

          <div className="constellation-controls graph-controls" aria-label="Graph view controls">
            <button type="button" onClick={() => zoomTo(transform.k * 0.82)} title="Zoom out">-</button>
            <input type="range" min="0.45" max="7" step="0.05" value={Number(transform.k.toFixed(2))} onChange={(event) => zoomTo(Number(event.target.value))} aria-label="Graph zoom level" />
            <button type="button" onClick={() => zoomTo(transform.k * 1.22)} title="Zoom in">+</button>
            <button type="button" onClick={() => applyTransform(zoomIdentity)} title="Fit graph">Fit</button>
            <button type="button" onClick={clearFocus} title="Clear hover or pinned focus">Clear</button>
          </div>

          <div className="graph-canvas-wrap">
            <svg ref={svgRef} className="graph-svg graph-svg-hero" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="xMidYMid meet">
              <defs>
                <radialGradient id="graph-bg-glow" cx="50%" cy="48%" r="66%">
                  <stop offset="0%" stopColor="rgba(245, 158, 11, 0.12)" />
                  <stop offset="52%" stopColor="rgba(14, 165, 233, 0.06)" />
                  <stop offset="100%" stopColor="rgba(15, 23, 42, 0)" />
                </radialGradient>
              </defs>
              <rect x="0" y="0" width={WIDTH} height={HEIGHT} className="graph-bg" />
              <circle cx={WIDTH / 2} cy={HEIGHT / 2} r="310" fill="url(#graph-bg-glow)" />
              <g className="graph-rings" transform={`translate(${WIDTH / 2},${HEIGHT / 2})`}>
                <circle r="170" />
                <circle r="260" />
                <circle r="338" />
              </g>
              <g transform={transform.toString()}>
                {communities.slice(0, 5).map((community, index) => (
                  <circle key={community.id || community.community_id || index} cx={WIDTH / 2} cy={HEIGHT / 2} r={120 + index * 46} className="community-bubble" />
                ))}
                {layout.edges.map((edge, index) => {
                  const source = edgeSource(edge)
                  const target = edgeTarget(edge)
                  const highlighted = activeId && (source === activeId || target === activeId)
                  const dimmed = activeId && !highlighted
                  const midX = (edge.sourceNode.x + edge.targetNode.x) / 2
                  const midY = (edge.sourceNode.y + edge.targetNode.y) / 2
                  const curve = Math.abs(edge.sourceNode.x - edge.targetNode.x) > 220 ? 34 : 12
                  return (
                    <path key={`${source}-${target}-${edge.type}-${index}`}
                      d={`M${edge.sourceNode.x},${edge.sourceNode.y} Q${midX},${midY - curve} ${edge.targetNode.x},${edge.targetNode.y}`}
                      stroke={highlighted ? '#f59e0b' : edgeColor(edge.type)}
                      className={`graph-edge ${dimmed ? 'dimmed' : ''} ${highlighted ? 'active retrieval-trail' : ''}`}
                      strokeWidth={highlighted ? 3 : Math.max(1, Math.min(2.4, Number(edge.weight || edge.score || 1)))} />
                  )
                })}
                {layout.nodes.map((node) => {
                  const dimmed = activeId && !connectedIds.has(node.id)
                  const focused = connectedIds.has(node.id)
                  const selected = selectedId === node.id
                  return (
                    <g key={node.id}
                      className={`graph-node ${node.type} ${dimmed ? 'dimmed' : ''} ${focused ? 'focused' : ''} ${selected ? 'selected' : ''}`}
                      onMouseEnter={() => setHoverId(node.id)}
                      onMouseLeave={() => setHoverId(null)}
                      onClick={() => setSelectedId((current) => current === node.id ? null : node.id)}>
                      <circle cx={node.x} cy={node.y} r={nodeRadius(node.type, node.degree)} fill={nodeColor(node.type)} />
                      <text x={node.x + nodeRadius(node.type, node.degree) + 7} y={node.y + 4}>{displayLabel(node.label, node.type === 'entity' ? 24 : 30)}</text>
                    </g>
                  )
                })}
              </g>
            </svg>
            <div className="graph-hover-card">
              {activeNode ? (
                <>
                  <strong>{activeNode.label}</strong>
                  <span>{activeNode.type}{activeNode.page_number ? ` | page ${activeNode.page_number}` : ''} | {Math.max(0, connectedIds.size - 1)} links</span>
                  <p>{activeNode.preview || activeNode.text || 'Connected graph evidence node.'}</p>
                </>
              ) : (
                <>
                  <strong>Graph focus</strong>
                  <span>Hover or click a node</span>
                  <p>Unrelated evidence fades back so the active relationship neighborhood is easier to read.</p>
                </>
              )}
            </div>
          </div>

          <div className="graph-legend">
            {NODE_TYPES.map((type) => (
              <button key={type} type="button" className={visibleTypes.has(type) ? 'active' : ''} onClick={() => toggleType(type)}>
                <i className={type} />{type}
              </button>
            ))}
          </div>
        </article>

        <article className="viz-card graph-path-card">
          <h4>Answer Subgraph Path Explorer</h4>
          <div className="graph-path-explorer">
            <div className="graph-path-column">
              <span>Query Entities</span>
              {(entityMatches.length ? entityMatches : (queryResult?.matched_entities || []).map((entity) => ({ entity }))).slice(0, 8).map((item, index) => (
                <strong key={`${item.entity || item.entity_id}-${index}`}>{item.entity || item.entity_id}</strong>
              ))}
              {!entityMatches.length && !(queryResult?.matched_entities || []).length && <em>No entity matches</em>}
            </div>
            <svg viewBox="0 0 760 270" className="graph-path-svg" preserveAspectRatio="xMidYMid meet">
              <defs>
                <marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
                </marker>
              </defs>
              {relationshipPaths.slice(0, 8).map((path, index) => {
                const y = 32 + index * 30
                const widthStroke = Math.max(1.5, Math.min(8, Number(path.score || 1)))
                return (
                  <g key={`${path.entity}-${path.related_entity}-${path.section}-${index}`} className="graph-path-row-svg">
                    <circle cx="90" cy={y} r="7" />
                    <text x="106" y={y + 4}>{displayLabel(path.entity || 'query', 18)}</text>
                    <path d={`M230,${y} C310,${y - 18} 390,${y + 18} 470,${y}`} strokeWidth={widthStroke} />
                    <text x="300" y={y - 8}>{displayLabel(path.edge_type || 'related', 18)}</text>
                    <circle cx="500" cy={y} r="7" />
                    <text x="516" y={y + 4}>{displayLabel(path.related_entity || path.section || 'evidence', 22)}</text>
                  </g>
                )
              })}
              {!relationshipPaths.length && <text x="380" y="136" textAnchor="middle" className="graph-path-empty">No relationship paths returned</text>}
            </svg>
            <div className="graph-path-column evidence">
              <span>Retrieved Evidence</span>
              {results.slice(0, 6).map((result) => (
                <strong key={result.chunk_id}>#{result.rank} {result.section_label || `P${result.page_number}`}</strong>
              ))}
            </div>
          </div>
          <div className="graph-community-strip">
            {(communityHits.length ? communityHits : communitySummaries).slice(0, 6).map((item) => (
              <span key={item.community_id || item.title}>{item.label || item.title || item.community_id}</span>
            ))}
            {!communityHits.length && !communitySummaries.length && <span>No community hits</span>}
          </div>
        </article>

        <article className="viz-card graph-score-card">
          <h4>Graph Score Stacks</h4>
          <div className="graph-score-stack-list">
            {results.slice(0, 8).map((result) => {
              const entries = scoreEntries(result.score_breakdown)
              const total = Math.max(1, entries.reduce((sum, row) => sum + row.value, 0))
              return (
                <div key={`${result.chunk_id}-stack`} className="graph-score-stack-row">
                  <strong>#{result.rank} {result.section_label || `P${result.page_number}`}</strong>
                  <div className="graph-score-stack-bar">
                    {entries.length ? entries.map((entry) => (
                      <span key={entry.key} className={`score-${entry.key}`} style={{ width: `${(entry.value / total) * 100}%` }} title={`${entry.label}: ${entry.value.toFixed(2)}`} />
                    )) : <span className="score-empty" style={{ width: '100%' }} />}
                  </div>
                  <small>{entries.map((entry) => `${entry.label} ${entry.value.toFixed(1)}`).join(' | ') || 'seed score only'}</small>
                </div>
              )
            })}
            {!results.length && <div className="comparison-empty">No scored evidence returned.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Query Path Flow</h4>
          <div className="query-path-flow compact-scroll">
            {pathRows.length ? pathRows.slice(0, 10).map((path, index) => (
              <div key={`${path.entity}-${path.section}-${index}`} className="query-path-flow-row">
                <span>{index + 1}</span>
                <strong>{path.entity || 'fallback'}</strong>
                <i>{path.related_entity ? `via ${path.related_entity}` : path.edge_type}</i>
                <strong>{path.section || 'section match'}</strong>
              </div>
            )) : <div className="comparison-empty">No query path flow returned.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Community Summaries</h4>
          <div className="graph-evidence-list compact-scroll">
            {communitySummaries.length ? communitySummaries.slice(0, 5).map((item) => (
              <div key={item.community_id || item.id || item.title} className="graph-evidence-card">
                <strong>{item.title || `Community ${item.community_id}`}</strong>
                <p>{item.summary}</p>
              </div>
            )) : <div className="comparison-empty">No community summaries returned for this graph.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Community Hits</h4>
          <div className="query-path-flow compact-scroll">
            {communityHits.length ? communityHits.slice(0, 8).map((item, index) => (
              <div key={`${item.community_id}-${index}`} className="query-path-flow-row">
                <span>{index + 1}</span>
                <strong>{item.label}</strong>
                <i>{item.score}</i>
                <strong>{(item.matched_entities || []).slice(0, 2).join(', ') || `${item.used_section_count} sections`}</strong>
              </div>
            )) : <div className="comparison-empty">No query-relevant graph communities for this answer.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Extracted Relationships</h4>
          <div className="query-path-flow compact-scroll">
            {relationships.length ? relationships.slice(0, 10).map((item, index) => (
              <div key={`${item.source_entity}-${item.relationship}-${item.target_entity}-${index}`} className="query-path-flow-row">
                <span>{index + 1}</span>
                <strong>{item.source_entity}</strong>
                <i>{item.relationship}</i>
                <strong>{item.target_entity}</strong>
              </div>
            )) : <div className="comparison-empty">No relationship triples returned for this graph.</div>}
          </div>
        </article>

        <article className="viz-card">
          <h4>Relationship Paths</h4>
          <div className="query-path-flow compact-scroll">
            {relationshipPaths.length ? relationshipPaths.slice(0, 10).map((path, index) => (
              <div key={`${path.entity}-${path.related_entity}-${path.section}-${index}`} className="query-path-flow-row">
                <span>{path.path_depth ?? index + 1}</span>
                <strong>{path.entity || 'query'}</strong>
                <i>{path.edge_type} | {path.score?.toFixed?.(2) ?? path.score}</i>
                <strong>{path.related_entity || path.section}</strong>
              </div>
            )) : <div className="comparison-empty">No relationship paths contributed to this answer.</div>}
          </div>
        </article>

        <article className="viz-card graph-evidence-wide">
          <h4>Cited Graph Evidence</h4>
          <div className="graph-evidence-list">
            {results.map((result) => (
              <div key={result.chunk_id} className="graph-evidence-card">
                <strong>#{result.rank} {result.section_label} | {result.graph_score?.toFixed?.(2) ?? result.graph_score}</strong>
                <div className="graph-entity-tags">
                  {(result.matched_entities || []).slice(0, 6).map((entity) => <span key={entity}>{entity}</span>)}
                </div>
                {result.score_breakdown && (
                  <div className="graph-entity-tags">
                    {Object.entries(result.score_breakdown).filter(([key]) => key !== 'details').slice(0, 5).map(([key, value]) => <span key={key}>{key}: {Number(value).toFixed?.(1) ?? value}</span>)}
                  </div>
                )}
                <p>{result.chunk_text_preview}</p>
              </div>
            ))}
            {!results.length && <div className="comparison-empty">No cited graph evidence returned.</div>}
          </div>
        </article>
      </div>
    </section>
  )
}
