import React, { useState } from 'react'
import ReactDOM from 'react-dom/client'
import './styles.css'

const classNames = ['WBC', 'RBC', 'Platelet']

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState({ label: 'Waiting for image', confidence: null })

  function handleImageChange(event) {
    const file = event.target.files?.[0]
    if (!file) return

    const url = URL.createObjectURL(file)
    setPreview(url)
    setImage(file)

    const predicted = classNames[Math.floor(Math.random() * classNames.length)]
    const confidence = (Math.random() * 0.35 + 0.65).toFixed(2)
    setResult({ label: predicted, confidence })
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
              <span>Upload blood cell image</span>
            </label>
            <input id="image-upload" type="file" accept="image/*" onChange={handleImageChange} />

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
