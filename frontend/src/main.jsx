import React, { useState } from 'react'
import ReactDOM from 'react-dom/client'
import './styles.css'

const classNames = ['WBC', 'RBC', 'Platelet']

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState('')
  const [threshold, setThreshold] = useState('0.6')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState({
    label: 'Waiting for image',
    confidence: null,
    probabilities: {},
    warning: '',
  })

  async function handleImageChange(event) {
    const file = event.target.files?.[0]
    if (!file) return

    const url = URL.createObjectURL(file)
    setPreview(url)
    setImage(file)
    setLoading(true)
    setResult({ label: 'Analyzing...', confidence: null, probabilities: {}, warning: '' })

    const formData = new FormData()
    formData.append('file', file)
    formData.append('threshold', threshold)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}))
        throw new Error(errorJson.error || 'Prediction failed')
      }

      const payload = await response.json()
      setResult({
        label: payload.predicted_class || 'Unknown',
        confidence: payload.confidence != null ? Number(payload.confidence).toFixed(4) : null,
        probabilities: payload.probabilities || {},
        warning: payload.warning || '',
      })
    } catch (error) {
      setResult({
        label: 'Error',
        confidence: null,
        probabilities: {},
        warning: error.message || 'Could not reach the prediction API.',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-shell">
      <div className="app-card">
        <header className="topbar">
          <div>
            <p className="eyebrow">Medical AI Lab</p>
            <h1>Blood Cell Classifier</h1>
          </div>
          <span className="pill">EfficientNet-B0</span>
        </header>

        <main className="content-grid">
          <section className="upload-panel">
            <label className="upload-box" htmlFor="image-upload">
              <span className="upload-icon">⤴</span>
              <span>{loading ? 'Analyzing image...' : 'Upload blood cell image'}</span>
            </label>
            <input id="image-upload" type="file" accept="image/*" onChange={handleImageChange} />

            <div className="threshold-control">
              <label htmlFor="threshold">Confidence threshold</label>
              <input
                id="threshold"
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
              />
            </div>

            <div className="info-list">
              <div>
                <strong>Model:</strong> EfficientNet-B0
              </div>
              <div>
                <strong>Classes:</strong> WBC, RBC, Platelet
              </div>
              <div>
                <strong>Input:</strong> Single cropped cell image
              </div>
            </div>
          </section>

          <section className="result-panel">
            {preview ? (
              <>
                <div className="image-wrap">
                  <img src={preview} alt="Uploaded blood cell" />
                </div>
                <div className="result-box">
                  <p className="result-label">Prediction</p>
                  <h2>{result.label}</h2>
                  {result.confidence && <span className="confidence">Confidence: {result.confidence}</span>}
                  {result.warning && <div className="warning-box">{result.warning}</div>}
                  {Object.keys(result.probabilities).length > 0 && (
                    <ul className="probability-list">
                      {Object.entries(result.probabilities).map(([label, value]) => (
                        <li key={label}><span>{label}</span><strong>{Number(value).toFixed(4)}</strong></li>
                      ))}
                    </ul>
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">🧪</div>
                <p>No image selected yet</p>
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
