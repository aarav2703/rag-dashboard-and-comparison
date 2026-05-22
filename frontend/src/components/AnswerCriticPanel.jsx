import React from 'react'

const VERDICT_LABELS = {
  accepted: 'Grounded',
  rejected: 'Rejected',
  insufficient_evidence: 'Insufficient evidence',
}

function verdictClass(verdict) {
  if (verdict === 'accepted') return 'accepted'
  if (verdict === 'insufficient_evidence') return 'insufficient'
  return 'rejected'
}

export default function AnswerCriticPanel({ queryResult, title = 'Self-Healing Answer' }) {
  const results = queryResult?.results || []
  const critic = queryResult?.critic || {}
  const attempts = queryResult?.answer_attempts || []
  const verdict = critic.verdict || (queryResult?.answer ? 'accepted' : 'pending')
  const issues = critic.issues || []
  const confidence = typeof critic.confidence === 'number' ? critic.confidence : null

  return (
    <div className="answer-critic-panel">
      <div className="answer-critic-head">
        <div>
          <h4>{title}</h4>
          <span>{queryResult?.answer_source || 'awaiting answer'} | {queryResult?.answer_model || 'no model'}</span>
        </div>
        <strong className={`critic-verdict ${verdictClass(verdict)}`}>
          {VERDICT_LABELS[verdict] || 'Pending'}
        </strong>
      </div>

      <p className="answer-critic-text">
        {queryResult?.answer || results[0]?.chunk_text_preview || 'No answer available yet.'}
      </p>

      <div className="answer-critic-meta">
        <span>Evidence {queryResult?.evidence_count ?? results.length}</span>
        <span>{attempts.length || 1} attempt{(attempts.length || 1) === 1 ? '' : 's'}</span>
        <span>{critic.retry_used ? 'retry used' : 'no retry'}</span>
        {confidence !== null && <span>{Math.round(confidence * 100)}% confidence</span>}
      </div>

      {critic.retry_query && (
        <div className="answer-critic-retry">
          <span>Retry query</span>
          <p>{critic.retry_query}</p>
        </div>
      )}

      {issues.length > 0 && (
        <div className="answer-critic-issues">
          {issues.slice(0, 3).map((issue, index) => (
            <span key={`${issue}-${index}`}>{issue}</span>
          ))}
        </div>
      )}
    </div>
  )
}
