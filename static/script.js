// Remove the import statement - Chart.js should be loaded via CDN in HTML

class BrainDiagnosisApp {
  constructor() {
    this.initializeElements()
    this.attachEventListeners()
    this.chart = null
    this.currentFile = null
  }

  initializeElements() {
    // Form elements
    this.uploadForm = document.getElementById("uploadForm")
    this.inputImage = document.getElementById("inputImage")
    this.dropzone = document.getElementById("dropzone")
    this.submitBtn = document.getElementById("submitBtn")
    this.submitText = document.getElementById("submitText")
    this.spinner = document.getElementById("spinner")

    // File info elements
    this.fileInfo = document.getElementById("fileInfo")
    this.fileName = document.getElementById("fileName")
    this.removeFileBtn = document.getElementById("removeFile")

    // Progress elements
    this.progressContainer = document.getElementById("progressContainer")
    this.progressFill = document.getElementById("progressFill")
    this.progressText = document.getElementById("progressText")

    // Results elements
    this.results = document.getElementById("results")
    this.previewImage = document.getElementById("previewImage")
    this.resultFileName = document.getElementById("resultFileName")
    this.classification = document.getElementById("classification")
    this.confidenceBadge = document.getElementById("confidenceBadge")
    this.probabilityBars = document.getElementById("probabilityBars")

    // Error elements
    this.errorMessage = document.getElementById("errorMessage")
    this.errorText = document.getElementById("errorText")
  }

  attachEventListeners() {
    // File input change
    this.inputImage.addEventListener("change", (e) => this.handleFileSelect(e))

    // Drag and drop
    this.dropzone.addEventListener("dragover", (e) => this.handleDragOver(e))
    this.dropzone.addEventListener("dragleave", (e) => this.handleDragLeave(e))
    this.dropzone.addEventListener("drop", (e) => this.handleDrop(e))
    this.dropzone.addEventListener("click", () => this.inputImage.click())

    // Remove file
    this.removeFileBtn.addEventListener("click", (e) => this.removeFile(e))

    // Form submission
    this.uploadForm.addEventListener("submit", (e) => this.handleSubmit(e))
  }

  handleFileSelect(event) {
    const file = event.target.files[0]
    if (file) {
      this.setFile(file)
    }
  }

  handleDragOver(event) {
    event.preventDefault()
    this.dropzone.classList.add("dragover")
  }

  handleDragLeave(event) {
    event.preventDefault()
    this.dropzone.classList.remove("dragover")
  }

  handleDrop(event) {
    event.preventDefault()
    this.dropzone.classList.remove("dragover")

    const files = event.dataTransfer.files
    if (files.length > 0) {
      this.inputImage.files = files
      this.setFile(files[0])
    }
  }

  setFile(file) {
    // Validate file type
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/bmp", "image/tiff"]
    if (!allowedTypes.includes(file.type)) {
      this.showError("Invalid file type. Please upload an image file (JPEG, PNG, GIF, BMP, TIFF).")
      return
    }

    // Validate file size (16MB limit)
    const maxSize = 16 * 1024 * 1024 // 16MB
    if (file.size > maxSize) {
      this.showError("File too large. Please upload an image smaller than 16MB.")
      return
    }

    this.currentFile = file
    this.fileName.textContent = file.name

    // Show file info, hide dropzone content
    this.dropzone.querySelector(".dropzone-content").classList.add("hidden")
    this.fileInfo.classList.remove("hidden")

    // Enable submit button
    this.submitBtn.disabled = false

    this.hideError()
  }

  removeFile(event) {
    event.stopPropagation()
    this.currentFile = null
    this.inputImage.value = ""

    // Show dropzone content, hide file info
    this.dropzone.querySelector(".dropzone-content").classList.remove("hidden")
    this.fileInfo.classList.add("hidden")

    // Disable submit button
    this.submitBtn.disabled = true

    this.hideResults()
  }

  async handleSubmit(event) {
    event.preventDefault()

    if (!this.currentFile) {
      this.showError("Please select a brain scan image first.")
      return
    }

    this.setLoadingState(true)
    this.hideError()
    this.hideResults()

    const formData = new FormData()
    formData.append("brain_scan", this.currentFile)

    try {
      // Simulate progress
      this.showProgress()

      const response = await fetch("https://parkinson-no1w.onrender.com/api/classify", {
        method: "POST",
        body: formData,
      })

      const result = await response.json()

      if (response.ok && result.success) {
        this.displayResults(result.data)
      } else {
        this.showError(result.error || "Failed to analyze the image. Please try again.")
      }
    } catch (error) {
      console.error("Error:", error)
      this.showError("Network error. Please check your connection and try again.")
    } finally {
      this.setLoadingState(false)
      this.hideProgress()
    }
  }

  setLoadingState(loading) {
    this.submitBtn.disabled = loading

    if (loading) {
      this.submitText.classList.add("hidden")
      this.spinner.classList.remove("hidden")
    } else {
      this.submitText.classList.remove("hidden")
      this.spinner.classList.add("hidden")
    }
  }

  showProgress() {
    this.progressContainer.classList.remove("hidden")

    // Simulate progress animation
    let progress = 0
    const interval = setInterval(() => {
      progress += Math.random() * 15
      if (progress > 90) progress = 90

      this.progressFill.style.width = `${progress}%`

      if (progress > 30) {
        this.progressText.textContent = "Processing neural patterns..."
      }
      if (progress > 60) {
        this.progressText.textContent = "Analyzing brain regions..."
      }
      if (progress > 80) {
        this.progressText.textContent = "Generating diagnosis..."
      }
    }, 200)

    // Store interval to clear it later
    this.progressInterval = interval
  }

  hideProgress() {
    if (this.progressInterval) {
      clearInterval(this.progressInterval)
    }

    // Complete the progress bar
    this.progressFill.style.width = "100%"

    setTimeout(() => {
      this.progressContainer.classList.add("hidden")
      this.progressFill.style.width = "0%"
      this.progressText.textContent = "Analyzing image..."
    }, 500)
  }

  displayResults(data) {
    // Set image preview
    this.previewImage.src = URL.createObjectURL(this.currentFile)
    this.resultFileName.textContent = this.currentFile.name

    // Set classification
    this.classification.textContent = data.classification
    this.confidenceBadge.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`

    // Create probability bars
    this.createProbabilityBars(data.probabilities)

    // Create chart
    this.createChart(data.probabilities)

    // Show results
    this.results.classList.remove("hidden")
    this.results.scrollIntoView({ behavior: "smooth" })
  }

  createProbabilityBars(probabilities) {
    const diseases = [
      { key: "alzheimers", label: "Alzheimer's Disease", class: "alzheimers" },
      { key: "normal", label: "Normal", class: "normal" },
      { key: "parkinsons", label: "Parkinson's Disease", class: "parkinsons" },
    ]

    this.probabilityBars.innerHTML = diseases
      .map((disease) => {
        const percentage = (probabilities[disease.key] * 100).toFixed(1)
        return `
                <div class="probability-bar">
                    <div class="probability-label">
                        <span>${disease.label}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="probability-track">
                        <div class="probability-fill ${disease.class}" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `
      })
      .join("")
  }

  createChart(probabilities) {
    if (this.chart) {
      this.chart.destroy()
    }

    const ctx = document.getElementById("probChart").getContext("2d")

    this.chart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Alzheimer's Disease", "Normal", "Parkinson's Disease"],
        datasets: [
          {
            data: [probabilities.alzheimers * 100, probabilities.normal * 100, probabilities.parkinsons * 100],
            backgroundColor: ["rgba(239, 68, 68, 0.8)", "rgba(16, 185, 129, 0.8)", "rgba(245, 158, 11, 0.8)"],
            borderColor: ["rgba(239, 68, 68, 1)", "rgba(16, 185, 129, 1)", "rgba(245, 158, 11, 1)"],
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "white",
              padding: 20,
              font: {
                size: 12,
              },
            },
          },
          tooltip: {
            callbacks: {
              label: (context) => `${context.label}: ${context.parsed.toFixed(1)}%`,
            },
          },
        },
      },
    })
  }

  showError(message) {
    this.errorText.textContent = message
    this.errorMessage.classList.remove("hidden")

    // Auto-hide after 5 seconds
    setTimeout(() => {
      this.hideError()
    }, 5000)
  }

  hideError() {
    this.errorMessage.classList.add("hidden")
  }

  hideResults() {
    this.results.classList.add("hidden")
  }
}

// Global functions for button actions
let app // Declare the app variable

function downloadResults() {
  // Implementation for downloading results as PDF/report
  alert("Download functionality would be implemented here")
}

function resetForm() {
  app.removeFile(new Event("click"))
  app.hideResults()
  app.hideError()
  window.scrollTo({ top: 0, behavior: "smooth" })
}

function hideError() {
  app.hideError()
}

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  app = new BrainDiagnosisApp() // Assign the app variable
})

// Cleanup object URLs on page unload
window.addEventListener("unload", () => {
  if (app.previewImage && app.previewImage.src) {
    URL.revokeObjectURL(app.previewImage.src)
  }
})
