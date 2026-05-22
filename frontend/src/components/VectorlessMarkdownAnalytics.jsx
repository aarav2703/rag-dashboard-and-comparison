import React, { useEffect, useMemo, useRef, useState } from 'react'
import { hierarchy, select, tree as d3Tree, zoom, zoomIdentity } from 'd3'
import AnswerCriticPanel from './AnswerCriticPanel.jsx'

function confidenceColor(value) {
  if (value >= 0.75) return '#10b981'
  if (value >= 0.45) return '#f59e0b'
  if (value >= 0.18) return '#0ea5e9'
  return '#475569'
}

function truncate(label, length = 30) {
  if (!label) return ''
  return label.length > length ? `${label.slice(0, length - 1)}...` : label
}

function cloneVisible(node, collapsedIds) {
  const cloned = { ...node }
  if (collapsedIds.has(node.id)) {
    cloned.children = []; cloned._collapsedCount = node.children?.length || 0
  } else {
    cloned.children = (node.children || []).map((child) => cloneVisible(child, collapsedIds))
  }
  return cloned
}

export default function VectorlessMarkdownAnalytics({ queryResult, visData }) {
  const [collapsedIds, setCollapsedIds] = useState(new Set())
  const [hoverId, setHoverId] = useState(null)
  const [transform, setTransform] = useState(zoomIdentity)
  const svgRef = useRef(null)
  const treeData = visData?.tree
  const selectedPathIds = new Set(visData?.selected_path_ids || [])
  const selectedPath = visData?.selected_path || queryResult?.selected_path || []
  const heatmap = visData?.section_heatmap || queryResult?.section_confidence || []
  const results = queryResult?.results || []

  const width = 820; const height = 620
  const layout = useMemo(() => {
    if (!treeData) return { nodes: [], links: [] }
    const visible = cloneVisible(treeData, collapsedIds)
    const root = hierarchy(visible)
    const layoutTree = d3Tree().nodeSize([34, 168])
    layoutTree(root)
    const nodes = root.descendants()
    const minX = Math.min(...nodes.map((node) => node.x), 0)
    const maxX = Math.max(...nodes.map((node) => node.x), 0)
    const xOffset = (height - (maxX - minX)) / 2 - minX
    return {
      nodes: nodes.map((node) => ({ ...node, sx: 44 + node.y, sy: xOffset + node.x })),
      links: root.links(),
    }
  }, [treeData, collapsedIds])

  function toggleNode(nodeId) {
    setCollapsedIds((prev) => {
      const next = new Set(prev)
      if (next.has(nodeId)) next.delete(nodeId); else next.add(nodeId)
      return next
    })
  }

  useEffect(() => {
    if (!svgRef.current) return
    const behavior = zoom().scaleExtent([0.55, 7]).on('zoom', (event) => setTransform(event.transform))
    select(svgRef.current).call(behavior)
    return () => { select(svgRef.current).on('.zoom', null) }
  }, [])

  const hoverNode = layout.nodes.find((node) => node.data.id === hoverId)

  if (!treeData) {
    return (
      <section className="panel vectorless-panel">
        <h3>Document Navigation Tree</h3>
        <div className="visual-empty"><strong>No document tree yet</strong><span>Run Vectorless Markdown RAG and ask a question to see structure-first navigation.</span></div>
      </section>
    )
  }

  return (
    <section className="panel vectorless-panel">
      <div className="vectorless-head">
        <div><h3>Document Navigation Tree</h3><p className="coverage-note">Vectorless RAG follows document structure: root to section to paragraph to answer.</p></div>
        <div className="vectorless-stats">
          <span>{queryResult?.tree_stats?.section_count || 0} sections</span>
          <span>{queryResult?.tree_stats?.paragraph_count || 0} paragraphs</span>
          <button type="button" onClick={() => setCollapsedIds(new Set())}>Expand all</button>
        </div>
      </div>

      <div className="vectorless-breadcrumb">
        {selectedPath.length ? selectedPath.map((item, index) => (
          <React.Fragment key={`${item.id}-${index}`}>
            <span>{item.label}</span>
            {index < selectedPath.length - 1 && <i>/</i>}
          </React.Fragment>
        )) : <span>No selected path yet</span>}
      </div>

      <div className="vectorless-layout">
        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <AnswerCriticPanel queryResult={queryResult} title="Vectorless Self-Healing Answer" />
        </article>

        <article className="viz-card vectorless-tree-card">
          <div className="vectorless-tree-wrap">
            <svg ref={svgRef} className="vectorless-tree-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
              <rect x="0" y="0" width={width} height={height} className="vectorless-bg" />
              <g transform={transform.toString()}>
                {layout.links.map((link) => {
                  const isSelected = selectedPathIds.has(link.source.data.id) && selectedPathIds.has(link.target.data.id)
                  return (
                    <path key={`${link.source.data.id}-${link.target.data.id}`}
                      d={`M${link.source.sx},${link.source.sy} C${(link.source.sx + link.target.sx) / 2},${link.source.sy} ${(link.source.sx + link.target.sx) / 2},${link.target.sy} ${link.target.sx},${link.target.sy}`}
                      className={isSelected ? 'vectorless-link selected' : 'vectorless-link'}
                      style={isSelected ? { animationDelay: '0s' } : {}} />
                  )
                })}
                {layout.nodes.map((node) => {
                  const confidence = node.data.confidence || 0
                  const isSelected = selectedPathIds.has(node.data.id)
                  const childCount = node.data.children?.length || 0
                  return (
                    <g key={node.data.id}
                      className={`vectorless-node ${isSelected ? 'selected' : ''}`}
                      transform={`translate(${node.sx},${node.sy})`}
                      onMouseEnter={() => setHoverId(node.data.id)}
                      onMouseLeave={() => setHoverId(null)}
                      onClick={() => toggleNode(node.data.id)}>
                      <circle r={node.data.type === 'document' ? 12 : node.data.type === 'section' ? 9 : 6} fill={confidenceColor(confidence)} />
                      <text x={14} y={4}>{truncate(node.data.label)}</text>
                      {node.data._collapsedCount > 0 && <text x={14} y={18} className="collapsed-count">+{node.data._collapsedCount} hidden</text>}
                      {childCount > 0 && !node.data._collapsedCount && <text x={14} y={18} style={{ fill: 'var(--muted)', fontSize: 8 }}>{childCount} children</text>}
                    </g>
                  )
                })}
              </g>
            </svg>
            {hoverNode && (
              <div className="vectorless-hover-card">
                <strong>{hoverNode.data.label}</strong>
                <span>{hoverNode.data.type} | confidence {(hoverNode.data.confidence || 0).toFixed(2)}</span>
                <p>{hoverNode.data.preview || 'Click to collapse or expand this branch.'}</p>
              </div>
            )}
          </div>
          <div className="zoom-hint">Wheel to zoom. Drag to pan. Click a node to collapse or expand its branch.</div>
        </article>

        <article className="viz-card">
          <h4>Selected Query Path</h4>
          <div className="vectorless-path-list">
            {selectedPath.map((item, index) => (
              <div key={`${item.id}-${index}`} className="vectorless-path-row">
                <span>{index + 1}</span>
                <div><strong>{item.label}</strong><small>{item.type} | confidence {(item.confidence || 0).toFixed(2)}</small></div>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Section Confidence Heatmap</h4>
          <div className="section-heatmap">
            {heatmap.slice(0, 36).map((section) => (
              <div key={section.id} className={section.is_selected ? 'selected' : ''}
                style={{ backgroundColor: confidenceColor(section.confidence), opacity: 0.25 + section.confidence * 0.75 }}
                title={`${section.label}: ${section.confidence}`}>
                <span>{truncate(section.label, 18)}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Structural Evidence</h4>
          <div className="vectorless-results">
            {results.map((result) => (
              <div key={result.chunk_id} className="vectorless-result-card">
                <strong>#{result.rank} {result.section_label}</strong>
                <span>confidence {(result.section_confidence || 0).toFixed(2)} | page {result.page_number}</span>
                <p>{result.chunk_text_preview}</p>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
