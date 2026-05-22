import React from 'react'

function consoleLineClass(log) {
  const message = String(log.message || '').toLowerCase()
  if (log.level === 'ok') return 'ok'
  if (log.level === 'error') return 'error'
  if (log.level === 'warn') return 'warn'
  if (message.includes('completed successfully') || message.includes('ready for queries') || message.includes('retrieved') || message.includes('accepted grounded')) return 'ok'
  if (message.startsWith('running ') || message.includes('pipeline starting') || message.includes('query received')) return 'running'
  return log.level || 'info'
}

export default function ConsolePanel({ logs }) {
  return (
    <section className="panel console-panel">
      <h3>Pipeline Console</h3>
      <div className="console-window">
        {logs.length === 0 ? (
          <div className="console-line muted">Waiting for activity...</div>
        ) : (
          logs.map((log, index) => (
            <div key={`${log.time}-${index}`} className={`console-line ${consoleLineClass(log)}`}>
              <span className="console-time">{log.time}</span>
              <span className="console-message">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </section>
  )
}
