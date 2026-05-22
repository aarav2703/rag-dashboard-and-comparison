import React, { useEffect, useMemo, useState } from 'react'
import AnswerCriticPanel from './AnswerCriticPanel.jsx'

function nodeColor(type) {
  if (type === 'question') return '#e2e8f0'
  if (type === 'query') return '#f59e0b'
  if (type === 'bridge') return '#7c3aed'
  if (type === 'answer') return '#10b981'
  return '#0ea5e9'
}

function scoreLabel(value) {
  return typeof value === 'number' ? value.toFixed(3) : value || '-'
}

export default function MultiHopRagAnalytics({ queryResult, visData }) {
  const [selectedNode, setSelectedNode] = useState(null)
  const [bridgeAnimStep, setBridgeAnimStep] = useState(0)
  const graph = visData?.reasoning_graph || { nodes: [], links: [] }
  const hops = visData?.hops || queryResult?.hops || []
  const hopTable = visData?.hop_table || []
  const results = queryResult?.results || []
  const bridgeTerms = visData?.bridge_terms || queryResult?.bridge_terms || []
  const width = 980
  const height = 360

  const layout = useMemo(() => {
    const nodes = graph.nodes.map((node, index) => ({
      ...node,
      x: 70 + index * ((width - 140) / Math.max(1, graph.nodes.length - 1)),
      y: index % 2 ? 230 : 112,
    }))
    const byId = new Map(nodes.map((node) => [node.id, node]))
    return {
      nodes,
      links: graph.links.map((link) => ({ ...link, sourceNode: byId.get(link.source), targetNode: byId.get(link.target) })).filter((link) => link.sourceNode && link.targetNode),
    }
  }, [graph])

  useEffect(() => {
    const interval = setInterval(() => {
      setBridgeAnimStep((prev) => (prev >= bridgeTerms.length * 2 ? 0 : prev + 1))
    }, 700)
    return () => clearInterval(interval)
  }, [bridgeTerms.length])

  const active = selectedNode ? layout.nodes.find((node) => node.id === selectedNode) : null
  const highlightedBridgeIdx = Math.floor(bridgeAnimStep / 2)

  if (!layout.nodes.length) {
    return (
      <section className="panel multihop-panel">
        <h3>Hop-by-Hop Reasoning Graph</h3>
        <div className="visual-empty"><strong>No multi-hop trace yet</strong><span>Run Multi-hop RAG and ask a question to see bridge terms, hop queries, and final evidence.</span></div>
      </section>
    )
  }

  return (
    <section className="panel multihop-panel">
      <div className="multihop-head">
        <div><h3>Hop-by-Hop Reasoning Graph</h3><p className="coverage-note">Multi-hop RAG is useful when the first passage gives you a clue, not the whole answer.</p></div>
        <div className="multihop-stats">
          <span>{queryResult?.hop_count || hops.length} hops</span>
          <span>{queryResult?.bridge_entity || 'no bridge'} bridge</span>
          <span>{results.length} final chunks</span>
        </div>
      </div>

      <div className="multihop-layout">
        <article className="viz-card" style={{ gridColumn: '1 / -1' }}>
          <AnswerCriticPanel queryResult={queryResult} title="Multi-hop Self-Healing Answer" />
        </article>

        <article className="viz-card multihop-graph-card">
          <h4>Question to Bridge to Answer</h4>
          <div className="multihop-graph-wrap">
            <svg className="multihop-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
              <rect x="0" y="0" width={width} height={height} className="multihop-bg" />
              {layout.links.map((link, index) => {
                const isBridgeLink = link.label === 'extract bridge'
                const bridgePulse = isBridgeLink && bridgeAnimStep % 2 === 0
                return (
                  <g key={`${link.source}-${link.target}-${index}`}>
                    <path
                      d={`M${link.sourceNode.x + 42},${link.sourceNode.y} C${(link.sourceNode.x + link.targetNode.x) / 2},${link.sourceNode.y} ${(link.sourceNode.x + link.targetNode.x) / 2},${link.targetNode.y} ${link.targetNode.x - 42},${link.targetNode.y}`}
                      className="multihop-link"
                      strokeWidth={bridgePulse ? 5 : 3}
                      stroke={bridgePulse ? '#f59e0b' : undefined} />
                    <text x={(link.sourceNode.x + link.targetNode.x) / 2} y={(link.sourceNode.y + link.targetNode.y) / 2 - 8}
                      className="multihop-link-label">{link.label}</text>
                  </g>
                )
              })}
              {layout.nodes.map((node) => (
                <g key={node.id}
                  className={`multihop-node ${selectedNode === node.id ? 'selected' : ''}`}
                  transform={`translate(${node.x},${node.y})`}
                  onMouseEnter={() => setSelectedNode(node.id)} onMouseLeave={() => setSelectedNode(null)}>
                  <circle r="42" fill={nodeColor(node.type)} />
                  <text>{node.label}</text>
                </g>
              ))}
            </svg>
            {active && (
              <div className="multihop-hover-card">
                <strong>{active.label}</strong><span>{active.type}</span>
                <p>{active.text || active.preview || 'Evidence node in the reasoning path.'}</p>
              </div>
            )}
          </div>
        </article>

        <article className="viz-card">
          <h4>Hop Queries</h4>
          <div className="hop-list">
            {hops.map((hop) => (
              <div key={hop.hop}>
                <span>Hop {hop.hop}</span>
                <strong>{hop.query}</strong>
                <p>{hop.purpose}</p>
                <div className="bridge-term-strip">
                  {(hop.bridge_terms || []).slice(0, 6).map((term, i) => (
                    <i key={`${hop.hop}-${term}`}
                      style={{ opacity: highlightedBridgeIdx >= i ? 1 : 0.4, transition: 'opacity 300ms ease' }}>{term}</i>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card">
          <h4>Bridge Evidence Table</h4>
          <div className="hop-table">
            <div className="hop-table-row hop-table-head">
              <span>Page</span><span>Hop 1</span><span>Hop 2</span><span>Role</span><span>Score</span>
            </div>
            {hopTable.slice(0, 10).map((row) => (
              <div key={row.chunk_id} className={`hop-table-row ${row.role}`}>
                <span>P{row.page_number}</span>
                <span>{row.hop1_rank ? `#${row.hop1_rank}` : '-'}</span>
                <span>{row.hop2_rank ? `#${row.hop2_rank}` : '-'}</span>
                <span>{row.role}</span>
                <span>{scoreLabel(row.score)}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="viz-card multihop-final-card">
          <h4>Final Multi-hop Evidence</h4>
          <div className="multihop-results">
            {results.map((result) => (
              <div key={result.chunk_id}>
                <strong>#{result.rank} P{result.page_number} | {scoreLabel(result.multihop_score)}</strong>
                <small>{result.hop_role} | bridge: {(result.bridge_hits || []).join(', ') || 'none'}</small>
                <p>{result.chunk_text_preview}</p>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
