import React, { useMemo, useState } from 'react'
import AnswerCriticPanel from './AnswerCriticPanel.jsx'

function highlightText(text, spans = []) {
  if (!text) return null
  if (!spans.length) return text
  const sorted = [...spans].sort((left, right) => left.start - right.start || left.end - right.end)
  const nodes = []; let cursor = 0
  sorted.forEach((span, index) => {
    const start = Math.max(0, Math.min(text.length, span.start))
    const end = Math.max(start, Math.min(text.length, span.end))
    if (start > cursor) nodes.push(text.slice(cursor, start))
    nodes.push(<mark key={`${span.term}-${index}-${start}-${end}`} className="bm25-highlight">{text.slice(start, end)}</mark>)
    cursor = end
  })
  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

function barWidth(value, maxValue) {
  if (!maxValue) return '0%'
  return `${Math.max(6, Math.round((value / maxValue) * 100))}%`
}

export default function Bm25Analytics({ chunks, queryResult, visData }) {
  const [selectedTerm, setSelectedTerm] = useState(null)
  const results = queryResult?.results || []
  const queryTerms = queryResult?.query_terms || visData?.query_terms || []
  const missingTerms = queryResult?.missing_query_terms || visData?.missing_query_terms || []
  const termStats = queryResult?.term_stats || visData?.term_stats || []
  const topContributions = queryResult?.top_result_term_contributions || visData?.top_result_term_contributions || []
  const warning = visData?.warning || (missingTerms.length ? `Missing query terms: ${missingTerms.join(', ')}` : '')

  const maxRarity = useMemo(() => Math.max(...termStats.map((entry) => entry.idf || 0), 0), [termStats])
  const maxContribution = useMemo(() => Math.max(...topContributions.map((entry) => entry.score || 0), 0), [topContributions])
  const matrixRows = useMemo(() => results.slice(0, 12).map((result) => {
    const matched = new Set(result.matched_terms || [])
    const counts = new Map()
    ;(result.highlight_spans || []).forEach((span) => counts.set(span.term, (counts.get(span.term) || 0) + 1))
    return { ...result, matched, counts }
  }), [results])
  const pageHeatmap = useMemo(() => {
    const byPage = new Map()
    results.forEach((result) => {
      const page = result.page_number || '?'
      const current = byPage.get(page) || { page, count: 0, score: 0 }
      current.count += 1; current.score += result.bm25_score || 0
      byPage.set(page, current)
    })
    return Array.from(byPage.values()).sort((left, right) => Number(left.page) - Number(right.page))
  }, [results])
  const maxPageScore = Math.max(...pageHeatmap.map((page) => page.score), 1)

  return (
    <section className="panel bm25-panel">
      <h3>BM25 Lexical Retrieval</h3>
      <div className="bm25-layout">
        <div className="bm25-column bm25-evidence-column">
          <article className="viz-card">
            <h4>Keyword Evidence Highlighter</h4>
            <div className="coverage-note">Exact query-term hits are highlighted inside each retrieved chunk.</div>
            <div className="bm25-results">
              {results.length === 0 ? (
                <div className="bm25-empty">Run a BM25 query to see exact lexical evidence.</div>
              ) : (
                results.map((result, idx) => {
                  const hasSelected = !selectedTerm || (result.matched_terms || []).includes(selectedTerm)
                  return (
                  <div key={result.chunk_id} className={`bm25-result-card ${hasSelected ? '' : 'dimmed'}`}
                    style={{ animation: `fadeInUp 400ms ease forwards`, animationDelay: `${idx * 0.08}s` }}>
                    <div className="bm25-result-head">
                      <strong>#{result.rank} page {result.page_number}</strong>
                      <span>BM25 {result.bm25_score?.toFixed?.(3) ?? result.bm25_score}</span>
                    </div>
                    <div className="bm25-highlight-text">
                      {highlightText(result.full_chunk_text || result.chunk_text_preview || '', result.highlight_spans || [])}
                    </div>
                    <div className="bm25-term-tags">
                      {(result.matched_terms || []).map((term) => (
                        <button key={`${result.chunk_id}-${term}`} type="button"
                          className={`bm25-term-tag ${selectedTerm === term ? 'active' : ''}`}
                          onClick={() => setSelectedTerm((c) => c === term ? null : term)}>{term}</button>
                      ))}
                    </div>
                  </div>
                )})
              )}
            </div>
          </article>
        </div>

        <div className="bm25-column bm25-insights-column">
          <article className="viz-card">
            <h4>Answer Comparison Panel</h4>
            <AnswerCriticPanel queryResult={queryResult} title="BM25 Answer" />
          </article>

          <article className="viz-card">
            <h4>Query-Term x Chunk Matrix</h4>
            <div className="coverage-note">Each cell shows whether a retrieved chunk contains a query term. Click a term badge to filter.</div>
            <div className="bm25-matrix">
              <div className="bm25-matrix-head">
                <span>Chunk</span>
                {queryTerms.slice(0, 8).map((term) => (
                  <button key={term} className={selectedTerm === term ? 'active' : ''}
                    onClick={() => setSelectedTerm((c) => c === term ? null : term)}>{term}</button>
                ))}
              </div>
              {matrixRows.map((row) => (
                <div key={row.chunk_id} className="bm25-matrix-row">
                  <span>#{row.rank} P{row.page_number}</span>
                  {queryTerms.slice(0, 8).map((term) => {
                    const count = row.counts.get(term) || 0
                    return <i key={`${row.chunk_id}-${term}`} className={count ? 'hit' : ''}
                      title={`${term}: ${count} hit${count === 1 ? '' : 's'}`}
                      style={{ opacity: count ? Math.min(1, 0.34 + count * 0.22) : 0.12 }} />
                  })}
                </div>
              ))}
            </div>
          </article>

          <article className="viz-card">
            <h4>Lexical Page Heatmap</h4>
            <div className="bm25-page-heatmap">
              {pageHeatmap.map((page) => (
                <div key={page.page} style={{ '--heat': page.score / maxPageScore }} title={`Page ${page.page}: ${page.count} retrieved chunk(s)`}>
                  <strong>P{page.page}</strong><span>{page.count}</span>
                </div>
              ))}
            </div>
          </article>

          <article className="viz-card">
            <h4>Term Rarity Bars</h4>
            <div className="coverage-note">BM25 rewards exact lexical overlap. Rare matched terms contribute more than common ones.</div>
            <div className="bm25-bars">
              {termStats.length === 0 ? (
                <div className="bm25-empty">No query terms available.</div>
              ) : (
                termStats.map((entry) => (
                  <div key={entry.term} className="bm25-bar-row">
                    <div className="bm25-bar-label">
                      <span>{entry.term}</span>
                      <small>df {entry.document_frequency} | idf {entry.idf?.toFixed?.(3) ?? entry.idf}</small>
                    </div>
                    <div className="bm25-bar-track">
                      <div className="bm25-bar-fill" style={{ width: barWidth(entry.idf || 0, maxRarity), transition: 'width 500ms ease' }} />
                    </div>
                  </div>
                ))
              )}
            </div>
          </article>

          <article className="viz-card">
            <h4>Matched-Token Contribution Chart</h4>
            <div className="coverage-note">This shows how much each matched query token contributed to the top-ranked chunk.</div>
            <div className="bm25-bars">
              {topContributions.length === 0 ? (
                <div className="bm25-empty">No matched contributions for the top result.</div>
              ) : (
                topContributions.map((entry) => (
                  <div key={entry.term} className="bm25-bar-row">
                    <div className="bm25-bar-label">
                      <span>{entry.term}</span>
                      <small>score {entry.score?.toFixed?.(3) ?? entry.score}</small>
                    </div>
                    <div className="bm25-bar-track">
                      <div className="bm25-bar-fill bm25-bar-fill-accent" style={{ width: barWidth(entry.score || 0, maxContribution), transition: 'width 500ms ease' }} />
                    </div>
                  </div>
                ))
              )}
            </div>
          </article>

          <article className="viz-card">
            <h4>Missing Query-Term Warning</h4>
            {missingTerms.length ? (
              <div className="bm25-warning">The following query terms were not found in any chunk: <strong>{missingTerms.join(', ')}</strong></div>
            ) : (
              <div className="bm25-ok">All query terms were present in at least one chunk.</div>
            )}
          </article>
        </div>
      </div>
    </section>
  )
}
