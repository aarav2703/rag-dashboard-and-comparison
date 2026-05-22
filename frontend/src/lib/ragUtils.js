function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
}

function hashToken(token, dimensions) {
  let h = 0
  for (let i = 0; i < token.length; i += 1) {
    h = (h * 33 + token.charCodeAt(i)) >>> 0
  }
  return h % dimensions
}

export function vectorizeText(text, dimensions = 48) {
  const vector = new Array(dimensions).fill(0)
  const tokens = tokenize(text)

  tokens.forEach((token) => {
    const idx = hashToken(token, dimensions)
    const sign = token.length % 2 === 0 ? 1 : -1
    vector[idx] += sign
  })

  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1
  return vector.map((value) => value / norm)
}

export function buildEmbeddings(chunks, dimensions = 48) {
  return chunks.map((chunk, index) => {
    const vector = vectorizeText(chunk.chunk_text || chunk.preview || '', dimensions)
    return {
      ...chunk,
      vector,
      x: vector[0] || 0,
      y: vector[1] || 0,
      z: vector[2] || 0,
      is_retrieved: index < 5
    }
  })
}
