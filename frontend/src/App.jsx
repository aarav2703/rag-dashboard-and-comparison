import React, { useEffect, useRef, useState } from 'react'
import EmbeddingConstellation from './components/EmbeddingConstellation.jsx'
import NaiveRagAnalytics from './components/NaiveRagAnalytics.jsx'
import HybridAnalytics from './components/HybridAnalytics.jsx'
import CragAnalytics from './components/RerankAnalytics.jsx'
import GraphRagAnalytics from './components/GraphRagAnalytics.jsx'
import AgenticRagAnalytics from './components/AgenticRagAnalytics.jsx'
import MethodComparison from './components/MethodComparison.jsx'
import ConsolePanel from './components/ConsolePanel.jsx'
import { buildEmbeddings } from './lib/ragUtils.js'

const API_BASE = 'http://localhost:5000'

const METHOD_ICONS = { naive: '\u2606', hybrid: '\u29C8', graph: '\u2B21', agentic: '\u2699', crag: '\u21C5', compare: '\u2981' }

const PIPELINES = {
  naive: {
    label: 'Naive Vector RAG',
    shortLabel: 'Naive',
    prefix: 'naive_rag',
    note: 'Semantic embedding retrieval with a 2D projection of chunks and query evidence.',
    evidenceMode: 'Semantic similarity'
  },
  hybrid: {
    label: 'Hybrid RAG',
    shortLabel: 'Hybrid',
    prefix: 'hybrid_rag',
    note: 'Combines semantic and lexical candidates, then rank-fuses the merged evidence pool.',
    evidenceMode: 'Vector + BM25 fusion'
  },
  graph: {
    label: 'GraphRAG',
    shortLabel: 'Graph',
    prefix: 'graph_rag',
    note: 'Builds a relationship graph, retrieves answer subgraphs, and highlights the evidence path toward full GraphRAG.',
    evidenceMode: 'Graph subgraph retrieval'
  },
  agentic: {
    label: 'Agentic Multi-hop RAG',
    shortLabel: 'Agent',
    prefix: 'agentic_rag',
    note: 'Plans tool use, extracts bridge clues, performs second-hop retrieval, and shows structured agent decisions.',
    evidenceMode: 'Agent tool + bridge loop'
  },
  crag: {
    label: 'Corrective RAG with Reranking',
    shortLabel: 'CRAG',
    prefix: 'crag_rag',
    note: 'Reranks candidate evidence, grades retrieval quality, rewrites or falls back when support is weak, then checks groundedness.',
    evidenceMode: 'Rerank + corrective grading'
  },
  compare: {
    label: 'Compare All Methods',
    shortLabel: 'Compare',
    prefix: '',
    note: 'Loads every method side by side for answer, evidence, and visual comparison.',
    evidenceMode: 'All methods',
    comparisonOnly: true
  }
}

const RUNNABLE_METHODS = Object.entries(PIPELINES)
  .filter(([, pipeline]) => !pipeline.comparisonOnly)
  .map(([id, pipeline]) => ({ id, ...pipeline }))

const INITIAL_METHOD_STATUS = {
  naive: 'red', hybrid: 'red', graph: 'red', agentic: 'red', crag: 'red'
}

function InfoButton({ text }) {
  return (
    <span className="info-popover">
      <button type="button" aria-label="Explain this RAG method">i</button>
      <span>{text}</span>
    </span>
  )
}

function PipelineBadge({ status }) {
  return (
    <span className={`pipeline-badge ${status}`}>
      <span className="pipeline-badge-light" />
      {status === 'red' && 'Not run'}
      {status === 'yellow' && 'Running'}
      {status === 'green' && 'Ready'}
    </span>
  )
}

function hasUsableMethodArtifact(payload) {
  const chunks = payload?.chunks || []
  const queryResult = payload?.queryResult || {}
  const results = queryResult?.results || []
  return chunks.length > 0 || results.length > 0 || Boolean(queryResult?.answer)
}

function WorkspaceSummary({ chunks, queryResult, visData, activePipeline, sourceName }) {
  const results = queryResult?.results || []
  const citedPages = new Set(results.map((result) => result.page_number).filter(Boolean))
  const queryTerms = visData?.query_terms || queryResult?.query_terms || []
  const missingTerms = visData?.missing_query_terms || queryResult?.missing_query_terms || []

  return (
    <section className="panel workspace-summary">
      <div className="summary-card">
        <span>Corpus</span>
        <strong>{chunks.length}</strong>
        <small>{sourceName}</small>
      </div>
      <div className="summary-card">
        <span>Retrieved</span>
        <strong>{results.length}</strong>
        <small>{activePipeline.evidenceMode}</small>
      </div>
      <div className="summary-card">
        <span>Cited Pages</span>
        <strong>{citedPages.size}</strong>
        <small>{citedPages.size ? `P${Array.from(citedPages).slice(0, 4).join(', P')}` : 'No evidence yet'}</small>
      </div>
      <div className="summary-card">
        <span>{queryTerms.length ? 'Query Terms' : 'Answer'}</span>
        <strong>{queryTerms.length || (queryResult?.answer ? 'Ready' : '-')}</strong>
        <small>{missingTerms.length ? `${missingTerms.length} missing terms` : queryResult?.answer_source || 'Awaiting query'}</small>
      </div>
    </section>
  )
}

function MethodTimeline({ activePipeline, queryResult }) {
  const retrieved = queryResult?.results?.length || 0
  const isCrag = queryResult?.mode === 'crag'
  const accepted = isCrag && queryResult?.critic?.verdict === 'accepted'
  const steps = [
    { label: 'Load Corpus', detail: activePipeline.shortLabel },
    { label: 'Retrieve', detail: retrieved ? `${retrieved} chunks` : 'waiting' },
    { label: isCrag ? 'Rerank' : 'Generate', detail: queryResult?.answer ? 'answer ready' : 'idle' },
    { label: isCrag ? 'Grade' : 'Finalize', detail: isCrag ? (queryResult?.critic?.verdict || queryResult?.crag_summary?.branch || 'pending') : (queryResult?.answer ? 'ready' : 'idle') },
    { label: 'Answer', detail: accepted ? 'grounded' : queryResult?.answer ? 'ready' : 'idle' }
  ]

  return (
    <section className="panel method-timeline-panel">
      <div>
        <h3>Method Timeline Trace</h3>
        <p>{activePipeline.label}</p>
      </div>
      <div className="timeline common-timeline">
        {steps.map((step, index) => {
          const isActive = Boolean(queryResult) && (index <= 1 || (index <= 3 && queryResult?.answer) || (isCrag && index === 4 && queryResult?.critic))
          return (
            <div key={step.label} className={`timeline-step ${isActive ? 'active' : ''}`}>
              <span className="timeline-dot" />
              <div><div className="timeline-name">{step.label}</div><div className="timeline-meta">{step.detail}</div></div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function SimpleAnswerPanel({ queryResult, title = 'Answer' }) {
  const results = queryResult?.results || []
  return (
    <div className="answer-critic-panel">
      <div className="answer-critic-head">
        <div>
          <h4>{title}</h4>
          <span>{queryResult?.answer_source || 'awaiting answer'} | {queryResult?.answer_model || 'no model'}</span>
        </div>
        <strong className="critic-verdict accepted">{queryResult?.answer ? 'Ready' : 'Pending'}</strong>
      </div>
      <p className="answer-critic-text">
        {queryResult?.answer || results[0]?.chunk_text_preview || 'No answer available yet.'}
      </p>
      <div className="answer-critic-meta">
        <span>Evidence {queryResult?.evidence_count ?? results.length}</span>
        <span>{results.length ? `Top page P${results[0]?.page_number || '-'}` : 'No retrieval yet'}</span>
      </div>
    </div>
  )
}

export default function App() {
  const [pipelineMode, setPipelineMode] = useState('naive')
  const [chunks, setChunks] = useState([])
  const [sourceName, setSourceName] = useState('(no file loaded)')
  const [queryResult, setQueryResult] = useState(null)
  const [visData, setVisData] = useState(null)
  const [question, setQuestion] = useState('')
  const [logs, setLogs] = useState([])
  const [uploadedFile, setUploadedFile] = useState(null)
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineStatus, setPipelineStatus] = useState('red')
  const [methodStatuses, setMethodStatuses] = useState(INITIAL_METHOD_STATUS)
  const [corpusReady, setCorpusReady] = useState(false)
  const [selectedFileNeedsBuild, setSelectedFileNeedsBuild] = useState(false)
  const [comparisonData, setComparisonData] = useState({})
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const eventSourceRef = useRef(null)

  function pushLog(message, level = 'info') {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    setLogs((prev) => [...prev.slice(-99), { time, message, level }])
  }

  const activePipeline = PIPELINES[pipelineMode] || PIPELINES.naive
  const activeStatus = activePipeline.comparisonOnly
    ? (RUNNABLE_METHODS.every((method) => methodStatuses[method.id] === 'green') ? 'green' : corpusReady ? 'yellow' : 'red')
    : (methodStatuses[pipelineMode] || 'red')

  function getArtifactUrl(name) {
    return `/data/${activePipeline.prefix}_${name}`
  }

  async function loadMethodArtifacts(method) {
    const pipeline = PIPELINES[method]
    const [chunksRes, queryRes, visRes] = await Promise.all([
      fetch(`/data/${pipeline.prefix}_chunks.json`),
      fetch(`/data/${pipeline.prefix}_query_result.json`),
      fetch(`/data/${pipeline.prefix}_vis.json`)
    ])

    if (!chunksRes.ok || !queryRes.ok || !visRes.ok) {
      throw new Error(`${pipeline.label} artifacts are not available yet`)
    }

    const [chunksJson, queryJson, visJson] = await Promise.all([
      chunksRes.json(), queryRes.json(), visRes.json()
    ])

    return {
      chunks: buildEmbeddings(chunksJson?.chunks || []),
      queryResult: queryJson || { query: '', results: [] },
      visData: visJson || {}
    }
  }

  async function loadComparisonArtifacts() {
    const entries = await Promise.all(
      RUNNABLE_METHODS.map(async (method) => {
        try {
          const payload = await loadMethodArtifacts(method.id)
          return [method.id, payload]
        } catch (error) {
          return [method.id, { error: String(error.message || error) }]
        }
      })
    )
    const nextComparisonData = Object.fromEntries(entries)
    setComparisonData(nextComparisonData)
    pushLog('Loaded comparison artifacts for all available methods', 'ok')
    return nextComparisonData
  }

  function hydrateSingleMethod(methodId, payload) {
    if (!payload || payload.error) return
    setChunks(payload.chunks || [])
    setQueryResult(payload.queryResult || { query: '', results: [] })
    setVisData(payload.visData || {})
    if (hasUsableMethodArtifact(payload)) {
      setMethodStatuses((prev) => ({ ...prev, [methodId]: 'green' }))
      setCorpusReady(true)
    }
  }

  async function buildSelectedPdfCorpus(mode = 'naive') {
    if (!uploadedFile) throw new Error('Please select a file first')

    setPipelineRunning(true)
    setPipelineStatus('yellow')
    setMethodStatuses(INITIAL_METHOD_STATUS)
    setCorpusReady(false)
    setChunks([])
    setQueryResult(null)
    setVisData(null)
    setComparisonData({})
    pushLog(`Building fresh corpus from ${uploadedFile.name}...`, 'info')

    const formData = new FormData()
    formData.append('file', uploadedFile)
    const uploadRes = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData })
    if (!uploadRes.ok) throw new Error('Upload failed: ' + (await uploadRes.text()))
    const uploadData = await uploadRes.json()
    pushLog(`File uploaded successfully: ${uploadData.filename}`, 'ok')

    const runRes = await fetch(`${API_BASE}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    })
    if (!runRes.ok) throw new Error('Pipeline start failed: ' + (await runRes.text()))

    let completed = false
    let attempts = 0
    while (!completed && attempts < 300) {
      const statusRes = await fetch(`${API_BASE}/api/status`)
      if (statusRes.ok) {
        const status = await statusRes.json()
        if (status.state === 'green') {
          completed = true
          break
        }
      }
      await new Promise(resolve => setTimeout(resolve, 500))
      attempts += 1
    }

    if (!completed) throw new Error('Pipeline timeout')
    pushLog('Fresh PDF corpus ready', 'ok')
    setCorpusReady(true)
    setSelectedFileNeedsBuild(false)
    setPipelineStatus('green')
    setSourceName(uploadedFile.name)
    await new Promise(resolve => setTimeout(resolve, 500))
  }

  useEffect(() => {
    const setupEventStream = () => {
      const es = new EventSource(`${API_BASE}/api/logs`)
      eventSourceRef.current = es
      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.message === 'heartbeat') return
          const level = data.level || 'info'
          const msg = data.message || ''
          if (msg) pushLog(msg, level)
        } catch (e) {}
      }
      es.onerror = () => { es.close(); setTimeout(setupEventStream, 2000) }
    }
    setupEventStream()
    return () => { eventSourceRef.current?.close() }
  }, [])

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`)
        if (res.ok) { const data = await res.json(); setPipelineStatus(data.state) }
      } catch (e) {}
    }
    const interval = setInterval(pollStatus, 500)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (activePipeline.comparisonOnly) return
    const payload = comparisonData[pipelineMode]
    if (payload && !payload.error) { hydrateSingleMethod(pipelineMode, payload); return }
    loadMethodArtifacts(pipelineMode)
      .then((artifacts) => hydrateSingleMethod(pipelineMode, artifacts))
      .catch(() => {})
  }, [pipelineMode])

  async function handleFileSelection(event) {
    const file = event.target.files?.[0]
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) { pushLog('Only PDF files are supported', 'error'); return }
    setUploadedFile(file)
    setSelectedFileNeedsBuild(true)
    setPipelineStatus('red')
    setMethodStatuses(INITIAL_METHOD_STATUS)
    setCorpusReady(false)
    setChunks([])
    setQueryResult(null)
    setVisData(null)
    setComparisonData({})
    setSourceName('(no file loaded)')
    pushLog(`File selected: ${file.name}`, 'info')
  }

  async function handleRunPipeline() {
    if (!uploadedFile) {
      if (activePipeline.comparisonOnly) { await loadComparisonArtifacts(); return }
      pushLog('Please select a file first', 'error')
      return
    }
    if (activePipeline.comparisonOnly) {
      try {
        await buildSelectedPdfCorpus('naive')
        const artifacts = await loadComparisonArtifacts()
        RUNNABLE_METHODS.forEach((method) => {
          if (hasUsableMethodArtifact(artifacts[method.id])) {
            setMethodStatuses((prev) => ({ ...prev, [method.id]: 'green' }))
          }
        })
      } catch (e) {
        pushLog(`Pipeline error: ${String(e.message || e)}`, 'error')
      } finally {
        setPipelineRunning(false)
      }
      return
    }

    setPipelineRunning(true)
    setMethodStatuses((prev) => ({ ...prev, [pipelineMode]: 'yellow' }))

    try {
      await buildSelectedPdfCorpus(pipelineMode)
        pushLog('Pipeline completed successfully!', 'ok')
        setMethodStatuses((prev) => ({ ...prev, [pipelineMode]: 'green' }))
        try {
          const [chunksRes, queryRes, visRes] = await Promise.all([fetch(getArtifactUrl('chunks.json')), fetch(getArtifactUrl('query_result.json')), fetch(getArtifactUrl('vis.json'))])
          const [chunksJson, queryJson, visJson] = await Promise.all([chunksRes.json(), queryRes.json(), visRes.json()])
          const loadedChunks = buildEmbeddings(chunksJson?.chunks || [])
          setChunks(loadedChunks); setQueryResult(queryJson || { query: '', results: [] }); setVisData(visJson || { points: [], query_point: null })
          setSourceName(uploadedFile.name)
        } catch (e) { pushLog(`Failed to load pipeline output: ${String(e)}`, 'error') }
    } catch (e) { pushLog(`Pipeline error: ${String(e)}`, 'error'); setMethodStatuses((prev) => ({ ...prev, [pipelineMode]: 'red' })) }
    finally { setPipelineRunning(false) }
  }

  async function handleAskQuestion() {
    if (!question.trim()) { pushLog('Type a question first.', 'error'); return }

    if (activePipeline.comparisonOnly) {
      if (uploadedFile && selectedFileNeedsBuild) {
        try {
          await buildSelectedPdfCorpus('naive')
        } catch (error) {
          setPipelineRunning(false)
          pushLog(`Pipeline error: ${String(error.message || error)}`, 'error')
          return
        } finally {
          setPipelineRunning(false)
        }
      }
      if (!corpusReady && pipelineStatus !== 'green') { pushLog('Build the PDF corpus first.', 'error'); return }
      pushLog(`Comparison question: "${question}"`, 'info')
      for (const method of RUNNABLE_METHODS) {
        setMethodStatuses((prev) => ({ ...prev, [method.id]: 'yellow' }))
        pushLog(`Running ${method.label}...`, 'info')
        try {
          const queryRes = await fetch(`${API_BASE}/api/query`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: question, mode: method.id }) })
          if (!queryRes.ok) throw new Error(await queryRes.text())
          const queryJson = await queryRes.json()
          const artifacts = await loadMethodArtifacts(method.id)
          const payload = { ...artifacts, queryResult: queryJson }
          setComparisonData((prev) => ({ ...prev, [method.id]: payload }))
          if (pipelineMode === method.id) hydrateSingleMethod(method.id, payload)
          setMethodStatuses((prev) => ({ ...prev, [method.id]: 'green' }))
          pushLog(`${method.label} complete: retrieved ${queryJson.results?.length || 0} chunks`, 'ok')
        } catch (error) {
          setComparisonData((prev) => ({ ...prev, [method.id]: { ...(prev[method.id] || {}), error: String(error.message || error) } }))
          setMethodStatuses((prev) => ({ ...prev, [method.id]: 'red' }))
          pushLog(`${method.label} failed: ${String(error.message || error)}`, 'error')
        }
      }
      return
    }

    if (!chunks.length) { pushLog('No corpus loaded. Run a pipeline first.', 'error'); return }
    if (!corpusReady && pipelineStatus !== 'green') { pushLog('Pipeline not ready.', 'error'); return }

    pushLog(`Question: "${question}"`, 'info')
    setMethodStatuses((prev) => ({ ...prev, [pipelineMode]: 'yellow' }))
    try {
      const queryRes = await fetch(`${API_BASE}/api/query`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: question, mode: pipelineMode }) })
      if (!queryRes.ok) throw new Error('Query failed: ' + (await queryRes.text()))
      const result = await queryRes.json()
      setQueryResult(result)
      const artifacts = await loadMethodArtifacts(pipelineMode)
      setVisData(artifacts.visData)
      if (pipelineMode === 'naive') {
        const retrievedIds = new Set(result.results.map(r => r.chunk_id))
        const updatedPoints = (artifacts.visData.points || []).map(p => ({ ...p, is_retrieved: retrievedIds.has(p.chunk_id) }))
        setVisData({ ...artifacts.visData, points: updatedPoints })
      }
      setMethodStatuses((prev) => ({ ...prev, [pipelineMode]: 'green' }))
      pushLog(`Retrieved ${result.results.length} chunks`, 'ok')
    } catch (e) { setMethodStatuses((prev) => ({ ...prev, [pipelineMode]: 'red' })); pushLog(`Query error: ${String(e)}`, 'error') }
  }

  useEffect(() => { pushLog('Ready. Upload a PDF and run a RAG mode to inspect retrieval evidence.', 'info') }, [])

  return (
    <div className="app-layout">
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          RAG Evidence<br />Lab
          <span>Retrieval Diagnostics</span>
        </div>

        <div>
          <div className="sidebar-section-label">Retrieval Methods</div>
          <nav className="sidebar-methods">
            {Object.entries(PIPELINES).map(([mode, pipeline]) => (
              <button
                key={mode}
                type="button"
                className={`sidebar-method-btn ${pipelineMode === mode ? 'active' : ''} ${methodStatuses[mode] || ''}`}
                onClick={() => setPipelineMode(mode)}
                disabled={pipelineRunning}
              >
                <span className="sidebar-method-icon">{METHOD_ICONS[mode]}</span>
                {pipeline.shortLabel}
                {!pipeline.comparisonOnly && (
                  <span className={`sidebar-status-dot ${methodStatuses[mode] || 'red'}`} aria-label={`${pipeline.shortLabel} status`} />
                )}
              </button>
            ))}
          </nav>
        </div>

        <div className="sidebar-section-label">Pipeline</div>
        <div style={{ display: 'grid', gap: 8 }}>
          <label className="file-control" style={{ fontSize: 10, color: 'var(--muted)', display: 'grid', gap: 4 }}>
            Upload PDF
            <input type="file" accept=".pdf" onChange={handleFileSelection} disabled={pipelineRunning} style={{ fontSize: 10, padding: '7px 8px' }} />
          </label>
          <button className="run-button" onClick={handleRunPipeline}
            disabled={(!uploadedFile && !activePipeline.comparisonOnly) || pipelineRunning}
            style={{ opacity: (!uploadedFile && !activePipeline.comparisonOnly) || pipelineRunning ? 0.5 : 1, fontSize: 11 }}>
            {pipelineRunning ? 'Running...' : activePipeline.comparisonOnly ? 'Build Fresh Corpus' : `Run ${activePipeline.shortLabel}`}
          </button>
          <PipelineBadge status={activeStatus} />
          {uploadedFile && <div style={{ fontSize: 10, color: 'var(--muted)', wordBreak: 'break-all' }}>{uploadedFile.name}</div>}
          {selectedFileNeedsBuild && <div style={{ fontSize: 10, color: 'var(--accent)' }}>Selected PDF has not been indexed yet.</div>}
        </div>

        <div className="sidebar-section-label">Query</div>
        <div style={{ display: 'grid', gap: 8 }}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={activePipeline.comparisonOnly ? 'Compare across methods...' : 'Ask a question...'}
            disabled={!activePipeline.comparisonOnly && !corpusReady && pipelineStatus !== 'green'}
            style={{ border: '1px solid var(--line)', borderRadius: 10, background: 'var(--card-alt)', color: 'var(--ink)', padding: '10px 12px', fontSize: 12, fontFamily: 'inherit' }}
          />
          <button className="ask-button" onClick={handleAskQuestion}
            disabled={!activePipeline.comparisonOnly && !corpusReady && pipelineStatus !== 'green'}
            style={{ opacity: !activePipeline.comparisonOnly && !corpusReady && pipelineStatus !== 'green' ? 0.5 : 1, fontSize: 11 }}>
            {activePipeline.comparisonOnly ? 'Compare All' : 'Search'}
          </button>
        </div>

        <div className="sidebar-footer">
          {sourceName}<br />
          {Object.entries(methodStatuses).filter(([, s]) => s === 'green').length} / {RUNNABLE_METHODS.length} methods ready
        </div>
      </aside>

      <main className="app-main">
        <header className="hero" style={{ marginBottom: 18 }}>
          <h2>{activePipeline.label}</h2>
          <p>{activePipeline.note}</p>
        </header>

        {!chunks.length && !activePipeline.comparisonOnly && (
          <div className="empty-state">
            <p>Ready to start</p>
            <span>Upload a PDF, pick a retrieval mode, and run the pipeline.</span>
          </div>
        )}

        {!activePipeline.comparisonOnly && (
          <WorkspaceSummary chunks={chunks} queryResult={queryResult} visData={visData} activePipeline={activePipeline} sourceName={sourceName} />
        )}

        {!activePipeline.comparisonOnly && (
          <MethodTimeline activePipeline={activePipeline} queryResult={queryResult} />
        )}

        <section className="workspace-topbar">
          <ConsolePanel logs={logs} />
        </section>

        {pipelineMode === 'compare' ? (
          <main className="compare-workspace">
            <MethodComparison methods={RUNNABLE_METHODS} comparisonData={comparisonData} evalData={null} />
          </main>
        ) : (
          <div className={`main project-layout ${pipelineMode}-page`}>
            <section className="main-visual">
              <div className="method-title-row">
                <h3>{activePipeline.label}</h3>
                <InfoButton text={activePipeline.note} />
              </div>
              {pipelineMode === 'graph' ? (
                <GraphRagAnalytics queryResult={queryResult} visData={visData} />
              ) : pipelineMode === 'agentic' ? (
                <AgenticRagAnalytics queryResult={queryResult} visData={visData} />
              ) : pipelineMode === 'crag' ? (
                <CragAnalytics queryResult={queryResult} visData={visData} />
              ) : pipelineMode === 'hybrid' ? (
                <HybridAnalytics queryResult={queryResult} visData={visData} />
              ) : (
                <>
                  <EmbeddingConstellation chunks={chunks} queryResult={queryResult} visData={visData} />
                  {pipelineMode === 'naive' && (
                    <div className="below-visual-answer">
                      <SimpleAnswerPanel queryResult={queryResult} title="Naive Vector Answer" />
                    </div>
                  )}
                </>
              )}
            </section>

            <aside className="main-analytics">
              {pipelineMode === 'graph' ? (
                <div className="panel graph-side-panel">
                  <h3>Graph Evidence Trail</h3>
                  <div className="graph-evidence-list">
                    {(queryResult?.path_explanation || []).slice(0, 8).map((path, index) => (
                      <div key={`${path.entity}-${path.section}-${index}`} className="graph-evidence-card">
                        <strong>{path.entity || 'Query'} -&gt; {path.related_entity || path.edge_type}</strong>
                        <p>{path.section || 'Evidence section'}</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : pipelineMode === 'crag' ? (
                <div className="panel rerank-evidence-panel">
                  <h3>Corrected Final Evidence</h3>
                  <div className="rerank-summary-grid" style={{ marginBottom: 12 }}>
                    <div><span>Branch</span><strong>{queryResult?.crag_summary?.branch || '-'}</strong></div>
                    <div><span>Action</span><strong>{queryResult?.crag_summary?.action || '-'}</strong></div>
                  </div>
                  <div className="hybrid-results">
                    {(queryResult?.results || []).map((result) => (
                      <div key={result.chunk_id} className="hybrid-result-row">
                        <span className={`source-badge ${result.movement_label}`}>{result.movement > 0 ? `+${result.movement}` : result.movement}</span>
                        <div>
                          <strong>#{result.rank} page {result.page_number} | {result.reranker_score?.toFixed?.(3) ?? result.reranker_score}</strong>
                          <p>{result.chunk_text_preview}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : pipelineMode === 'agentic' ? (
                <div className="panel agentic-side-panel">
                  <h3>Agent Decisions</h3>
                  <div className="agent-scratchpad compact">
                    {(queryResult?.scratchpad || []).map((line, index) => (
                      <div key={`${line}-${index}`}><span>{String(index + 1).padStart(2, '0')}</span><p>{line}</p></div>
                    ))}
                  </div>
                  <h3 style={{ marginTop: 12 }}>Rejected Evidence</h3>
                  <div className="agent-rejected-pile compact">
                    {(queryResult?.rejected_evidence || []).slice(0, 5).map((item) => (
                      <div key={item.chunk_id}><strong>P{item.page_number}</strong><span>{item.reason}</span></div>
                    ))}
                  </div>
                </div>
              ) : pipelineMode === 'hybrid' ? (
                <div className="panel hybrid-evidence-panel">
                  <h3>Hybrid Final Evidence</h3>
                  <div className="hybrid-results">
                    {(queryResult?.results || []).map((result) => (
                      <div key={result.chunk_id} className="hybrid-result-row">
                        <span className={`source-badge ${result.source}`}>{result.source}</span>
                        <div>
                          <strong>#{result.rank} page {result.page_number} | {result.hybrid_score?.toFixed?.(3) ?? result.hybrid_score}</strong>
                          <p>{result.chunk_text_preview}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <NaiveRagAnalytics chunks={chunks} queryResult={queryResult} visData={visData} />
              )}
            </aside>
          </div>
        )}
      </main>
    </div>
  )
}
