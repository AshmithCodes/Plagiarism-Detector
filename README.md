# 🔍 Plagiarism Detector & Report Generator

A robust, asynchronous web-based plagiarism detection system built with **Flask**, **Scikit-Learn**, and **ReportLab**. The application processes document uploads in the background, segments text into logical chunks, runs natural language comparisons across both local reference files and live web repositories, and outputs a publication-quality PDF report with an annotated document markup.

---

## ✨ Features

* **⚡ Asynchronous Job Processing:** Utilizes threading to handle computationally intensive text extraction and live web scraping in the background without blocking the web server UI.
* **📄 Multi-Format Text Extraction:** Fully parses `.pdf`, `.docx`, `.doc`, and `.txt` documents using `pdfplumber` and `python-docx`.
* **🌐 Hybrid Cross-Referencing Web Engines:**
  * Runs specialized **Wikipedia API** contextual phrase fetches.
  * Dynamically scrapes live **DuckDuckGo HTML results** to catch obscure web text.
* **🧠 NLP-Driven Text Similarity:** Uses `TfidfVectorizer` (with 1–2 ngrams and English stop-word filtering) combined with **Cosine Similarity** and sequence matching to calculate high-accuracy confidence scores.
* **📈 Dynamic PDF Report Generation:** Automatically builds a styled, color-coded A4 PDF assessment including an overview scale, matched-source URLs, and an annotated document view highlighting plagiarized segments in light red.

---

## 🛠️ Project Structure

```text
plagcheck/
├── app.py              # Main Flask application, background workers & NLP logic
├── requirements.txt    # Application dependencies
├── uploads/            # Temporary directory for uploaded input & reference files
└── reports/            # Output directory for generated PDF evaluation reports

<img width="835" height="1283" alt="{645A838C-2FDE-4005-B7C9-0EAC8DBA8D5E}" src="https://github.com/user-attachments/assets/567383d0-f3c1-435d-b019-ec3534868a2a" />
