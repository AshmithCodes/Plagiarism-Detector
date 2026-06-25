"""
Plagiarism Detector - Flask Web App
====================================
Run:  python app.py
Open: http://localhost:5000

Dependencies:
    pip install flask pdfplumber python-docx reportlab scikit-learn requests beautifulsoup4
"""

import os, re, time, json, difflib, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template

import pdfplumber
from docx import Document as DocxDoc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests as req
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ── in-memory job store ────────────────────────────────────────────────────
jobs = {}   # job_id -> { status, progress, message, result, report_path }

# ── TEXT EXTRACTION ────────────────────────────────────────────────────────

def extract_text(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        parts = []
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                t = pg.extract_text()
                if t: parts.append(t)
        return "\n".join(parts)
    elif ext in (".docx", ".doc"):
        doc = DocxDoc(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

# ── SEGMENTATION ───────────────────────────────────────────────────────────

def clean(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\x20-\x7E]', ' ', text)).strip()

def split_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in raw if len(s.strip()) > 25]

def make_chunks(sents, w=2):
    out = []
    for i in range(len(sents) - w + 1):
        out.append(" ".join(sents[i:i+w]))
    for s in sents:
        if len(s) > 100 and s not in out:
            out.append(s)
    return list(dict.fromkeys(out))

def get_ngrams(text, n=5):
    words = re.findall(r'\w+', text.lower())
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}

# ── TFIDF ──────────────────────────────────────────────────────────────────

def tfidf_sim(query, docs):
    if not docs: return []
    try:
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)
        mat = vec.fit_transform([query] + docs)
        return cosine_similarity(mat[0:1], mat[1:]).flatten().tolist()
    except:
        return [0.0]*len(docs)

# ── WEB SOURCES ────────────────────────────────────────────────────────────

def search_wikipedia(query):
    results = []
    try:
        r = req.get("https://en.wikipedia.org/w/api.php", timeout=8, params={
            "action":"query","list":"search","srsearch":query[:180],
            "srlimit":2,"format":"json","srprop":"snippet","origin":"*"
        }, headers={"User-Agent":"PlagCheck/1.0"})
        hits = r.json().get("query",{}).get("search",[])
        for h in hits:
            snippet = BeautifulSoup(h.get("snippet",""), "html.parser").get_text()
            title = h.get("title","")
            # fetch short extract
            r2 = req.get("https://en.wikipedia.org/w/api.php", timeout=8, params={
                "action":"query","titles":title,"prop":"extracts",
                "exintro":True,"explaintext":True,"format":"json"
            }, headers={"User-Agent":"PlagCheck/1.0"})
            pages = r2.json().get("query",{}).get("pages",{})
            text = ""
            for p in pages.values():
                text = p.get("extract","")[:2500]
            results.append({
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ','_')}",
                "text": text or snippet
            })
    except Exception as e:
        pass
    return results

def search_duckduckgo(query):
    results = []
    try:
        r = req.post("https://html.duckduckgo.com/html/",
                     data={"q": query[:180]}, timeout=10,
                     headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"})
        soup = BeautifulSoup(r.text, "html.parser")
        for res in soup.select(".result")[:3]:
            t = res.select_one(".result__title")
            s = res.select_one(".result__snippet")
            u = res.select_one(".result__url")
            title   = t.get_text(strip=True) if t else ""
            snippet = s.get_text(strip=True) if s else ""
            url     = u.get_text(strip=True) if u else ""
            if not url.startswith("http"): url = "https://"+url
            if title and snippet:
                results.append({"title":title,"url":url,"text":snippet})
    except:
        pass
    return results

# ── CHECK ONE CHUNK ────────────────────────────────────────────────────────

def check_chunk(chunk, local_corpus):
    c = clean(chunk)
    c_ngrams = get_ngrams(c, 5)
    best, best_src, mtype = 0, None, None

    if local_corpus:
        sims = tfidf_sim(c, [d["text"] for d in local_corpus])
        for i,s in enumerate(sims):
            if s > best:
                best = s; best_src = local_corpus[i]; mtype = "local"

    q = " ".join(c.split()[:12])

    for wr in search_wikipedia(q):
        wt = wr["text"]
        s  = (tfidf_sim(c,[wt]) or [0])[0]
        ov = len(c_ngrams & get_ngrams(wt,5))
        s  = min(1.0, s + (0.18 if ov>=2 else 0))
        if s > best:
            best = s; best_src = wr; mtype = "Wikipedia"

    for dr in search_duckduckgo(q):
        sn = dr["text"]
        s  = (tfidf_sim(c,[sn]) or [0])[0]
        r2 = difflib.SequenceMatcher(None, c.lower()[:300], sn.lower()).ratio()
        s  = max(s, r2*0.9)
        if s > best:
            best = s; best_src = dr; mtype = "web"

    confidence = min(100, int(best * 170))
    return {
        "chunk": chunk,
        "is_flagged": confidence >= 35,
        "confidence": confidence,
        "source_title": best_src["title"] if best_src else None,
        "source_url":   best_src["url"]   if best_src else None,
        "match_type":   mtype
    }

# ── SCORE ──────────────────────────────────────────────────────────────────

def compute_score(results):
    flagged = [r for r in results if r["is_flagged"]]
    if not results: return 0
    ratio    = len(flagged)/len(results)
    avg_conf = sum(r["confidence"] for r in flagged)/len(flagged) if flagged else 0
    return min(100, int(ratio*60 + (avg_conf/100)*40))

def score_meta(s):
    if s<=15: return "#27ae60","Original","No significant plagiarism detected"
    if s<=35: return "#f1c40f","Minor Similarity","A few phrases match online sources"
    if s<=55: return "#e67e22","Moderate","Several sections need citation or rewriting"
    if s<=75: return "#e74c3c","High","Substantial portions appear copied"
    return "#c0392b","Severe","Document is largely unoriginal"

# ── PDF REPORT ─────────────────────────────────────────────────────────────

def build_pdf(input_name, sentences, results, score, out_path):
    col, level, advice = score_meta(score)
    sc = HexColor(col)
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=2*cm,rightMargin=2*cm,
                            topMargin=2.2*cm,bottomMargin=2*cm)
    S = getSampleStyleSheet()
    def sty(n,**k): return ParagraphStyle(n,parent=S["Normal"],**k)
    tit = sty("t",fontName="Helvetica-Bold",fontSize=18,alignment=TA_CENTER,
               textColor=HexColor("#1a1a2e"),spaceAfter=4)
    sub = sty("s",fontName="Helvetica",fontSize=10,alignment=TA_CENTER,
               textColor=HexColor("#666"),spaceAfter=16)
    sec = sty("h",fontName="Helvetica-Bold",fontSize=12,
               textColor=HexColor("#1a1a2e"),spaceBefore=12,spaceAfter=6)
    bod = sty("b",fontName="Helvetica",fontSize=9,
               textColor=HexColor("#2d2d2d"),leading=14,spaceAfter=3,alignment=TA_JUSTIFY)
    plg = sty("p",fontName="Helvetica",fontSize=9,
               textColor=HexColor("#6b0000"),backColor=HexColor("#ffe8e8"),
               leading=14,spaceAfter=2,alignment=TA_JUSTIFY,
               borderPad=5,borderColor=HexColor("#cc2222"),borderWidth=0.8)
    ann = sty("a",fontName="Helvetica-Oblique",fontSize=7.5,
               textColor=HexColor("#aa0000"),backColor=HexColor("#fff5f5"),
               leading=11,leftIndent=18,spaceAfter=8,
               borderPad=4,borderColor=HexColor("#ffbbbb"),borderWidth=0.5)
    ftr = sty("f",fontName="Helvetica",fontSize=7,
               textColor=HexColor("#aaa"),alignment=TA_CENTER,spaceBefore=6)

    story = []
    story.append(Paragraph("PLAGIARISM DETECTION REPORT", tit))
    story.append(Paragraph(f"File: {input_name}", sub))
    story.append(HRFlowable(width="100%",thickness=0.8,color=HexColor("#ccc")))
    story.append(Spacer(1,0.4*cm))

    # score box
    sb = Table([[
        Paragraph(f'<font size="44" color="{col}"><b>{score}%</b></font>',
                  sty("sc",fontName="Helvetica-Bold",fontSize=44,alignment=TA_CENTER)),
        Paragraph(f'<font name="Helvetica-Bold" size="13" color="{col}">{level}</font><br/>'
                  f'<font name="Helvetica" size="9" color="#444">{advice}</font><br/><br/>'
                  f'<font name="Helvetica" size="8" color="#888">'
                  f'Segments: {len(results)} &nbsp;|&nbsp; Flagged: {sum(1 for r in results if r["is_flagged"])}</font>',
                  sty("sd",fontName="Helvetica",fontSize=9,leading=16))
    ]], colWidths=[4.5*cm,12.5*cm])
    sb.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),1.5,sc),("BACKGROUND",(0,0),(-1,-1),HexColor("#fafafa")),
        ("LINEAFTER",(0,0),(0,0),0.5,HexColor("#ddd")),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LEFTPADDING",(0,0),(-1,-1),14),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(sb); story.append(Spacer(1,0.5*cm))

    # scale
    story.append(Paragraph("PLAGIARISM SCALE", sec))
    ld = [["Range","Level","Interpretation"]]
    rows = [("0–15%","Original","Content is substantially original"),
            ("16–35%","Minor","Small sections match; add citations"),
            ("36–55%","Moderate","Rewrite or cite flagged passages"),
            ("56–75%","High","Significant revision required"),
            ("76–100%","Severe","Document is largely unoriginal")]
    fills = ["#e8f8ee","#fef9e7","#fdf0e0","#fde8e8","#f8d7d7"]
    tcols = ["#1e8449","#9a7d0a","#a04000","#922b21","#7b241c"]
    for r in rows: ld.append(list(r))
    lt = Table(ld, colWidths=[2.8*cm,3.8*cm,10.4*cm])
    ls = [("BACKGROUND",(0,0),(-1,0),HexColor("#1a1a2e")),
          ("TEXTCOLOR",(0,0),(-1,0),white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
          ("FONTSIZE",(0,0),(-1,-1),8.5),("ALIGN",(0,0),(1,-1),"CENTER"),
          ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("GRID",(0,0),(-1,-1),0.4,HexColor("#ccc")),
          ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
          ("LEFTPADDING",(0,0),(-1,-1),8)]
    for i,(bg,tc) in enumerate(zip(fills,tcols)):
        ls += [("BACKGROUND",(0,i+1),(-1,i+1),HexColor(bg)),
               ("TEXTCOLOR",(0,i+1),(1,i+1),HexColor(tc)),
               ("FONTNAME",(0,i+1),(1,i+1),"Helvetica-Bold")]
    lt.setStyle(TableStyle(ls))
    story.append(lt); story.append(Spacer(1,0.5*cm))

    # sources
    flagged = [r for r in results if r["is_flagged"]]
    if flagged:
        story.append(Paragraph("MATCHED SOURCES", sec))
        for idx,res in enumerate(flagged,1):
            url  = res.get("source_url") or "N/A"
            src  = res.get("source_title") or "Unknown"
            conf = res["confidence"]
            cc   = "#c0392b" if conf>=60 else "#e67e22"
            row  = [[
                Paragraph(f'<b>#{idx}</b> &nbsp; Confidence: <font color="{cc}">{conf}%</font>',
                          sty("sh",fontSize=8.5)),
                Paragraph(f'<b>Source:</b> {src}<br/>'
                          f'<font color="#1155cc" size="7.5">{url[:90]}{"…" if len(url)>90 else ""}</font>',
                          sty("sd2",fontSize=8,leading=12))
            ]]
            tbl = Table(row,colWidths=[3.5*cm,13.5*cm])
            tbl.setStyle(TableStyle([
                ("BOX",(0,0),(-1,-1),0.5,HexColor("#ffbbbb")),
                ("BACKGROUND",(0,0),(0,0),HexColor("#fff0f0")),
                ("BACKGROUND",(1,0),(1,0),HexColor("#fafafa")),
                ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
                ("LEFTPADDING",(0,0),(-1,-1),10),("VALIGN",(0,0),(-1,-1),"TOP"),
            ]))
            story.append(tbl); story.append(Spacer(1,0.18*cm))
    else:
        story.append(Paragraph(
            "✓  No plagiarism detected — document appears to be original.",
            sty("ok",fontName="Helvetica-Bold",fontSize=10,
                textColor=HexColor("#1e8449"),backColor=HexColor("#e8f8ee"),
                borderPad=10,borderColor=HexColor("#27ae60"),borderWidth=0.8)))

    story.append(Spacer(1,0.4*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=HexColor("#ddd")))
    story.append(Spacer(1,0.3*cm))
    story.append(Paragraph("ANNOTATED DOCUMENT", sec))
    story.append(Paragraph(
        '<font color="#cc2222">■ Red = Plagiarised</font>  '
        '<font color="#333">■ Normal = Original</font>',
        sty("leg",fontSize=8,textColor=HexColor("#555"),spaceAfter=10)))

    flagged_map = {}
    for res in flagged:
        cw = set(res["chunk"].lower().split())
        for sent in sentences:
            sw = set(sent.lower().split())
            if sw and len(cw&sw)/len(sw) > 0.55:
                if sent not in flagged_map or res["confidence"] > flagged_map[sent]["confidence"]:
                    flagged_map[sent] = res

    counter = 0
    for sent in sentences:
        safe = sent.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        if sent in flagged_map:
            counter += 1
            r = flagged_map[sent]
            src = r.get("source_title") or "Unknown"
            url = r.get("source_url") or "N/A"
            mt  = r.get("match_type","web")
            story.append(Paragraph(f"[{counter}] {safe}", plg))
            story.append(Paragraph(
                f"↳ [{counter}] {src} | {r['confidence']}% via {mt} | "
                f"{url[:70]}{'…' if len(url)>70 else ''}",ann))
        else:
            story.append(Paragraph(safe, bod))

    story.append(Spacer(1,0.8*cm))
    story.append(HRFlowable(width="100%",thickness=0.4,color=HexColor("#ddd")))
    story.append(Paragraph(
        f"PlagiarismDetector v1.0 — CSE Project | File: {input_name} | "
        f"Sentences: {len(sentences)} | Flagged: {len(flagged)}", ftr))
    doc.build(story)

# ── BACKGROUND WORKER ──────────────────────────────────────────────────────

def run_job(job_id, input_path, ref_paths, max_chunks):
    def upd(pct, msg):
        jobs[job_id].update({"progress": pct, "message": msg})

    try:
        upd(5,  "Extracting text from document...")
        raw       = extract_text(input_path)
        text      = clean(raw)
        upd(12, "Segmenting into sentences...")
        sentences = split_sentences(text)
        all_chunks= make_chunks(sentences, 2)
        step      = max(1, len(all_chunks)//max_chunks)
        chunks    = all_chunks[::step][:max_chunks]

        local_corpus = []
        for rp in ref_paths:
            try:
                rt = clean(extract_text(rp))
                local_corpus.append({"title": Path(rp).name, "url":"local", "text": rt})
            except: pass

        upd(18, f"Checking {len(chunks)} segments...")
        results = []
        for i, chunk in enumerate(chunks):
            pct = 18 + int((i/len(chunks))*72)
            upd(pct, f"Segment {i+1} / {len(chunks)}")
            results.append(check_chunk(chunk, local_corpus))
            time.sleep(0.25)

        upd(92, "Computing score...")
        score = compute_score(results)
        col, level, advice = score_meta(score)

        upd(95, "Building PDF report...")
        stem       = Path(input_path).stem
        report_path= os.path.join(REPORT_FOLDER, f"{stem}_report.pdf")
        build_pdf(Path(input_path).name, sentences, results, score, report_path)

        flagged = [r for r in results if r["is_flagged"]]
        jobs[job_id].update({
            "status":   "done",
            "progress": 100,
            "message":  "Complete!",
            "result": {
                "score":        score,
                "color":        col,
                "level":        level,
                "advice":       advice,
                "total":        len(results),
                "flagged_count":len(flagged),
                "report_file":  Path(report_path).name,
                "segments": [{
                    "chunk":        r["chunk"][:220],
                    "confidence":   r["confidence"],
                    "source_title": r.get("source_title"),
                    "source_url":   r.get("source_url"),
                    "match_type":   r.get("match_type"),
                } for r in flagged]
            }
        })
    except Exception as e:
        jobs[job_id].update({"status":"error","message": str(e)})

# ── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    import uuid
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"No file"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in (".pdf",".docx",".doc",".txt"):
        return jsonify({"error":"Unsupported file type"}), 400

    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    f.save(fpath)

    ref_paths  = []
    for rf in request.files.getlist("refs"):
        rext = Path(rf.filename).suffix.lower()
        if rext in (".pdf",".docx",".txt"):
            rname = f"{uuid.uuid4().hex}{rext}"
            rpath = os.path.join(UPLOAD_FOLDER, rname)
            rf.save(rpath)
            ref_paths.append(rpath)

    max_chunks = int(request.form.get("max_chunks", 20))
    job_id     = uuid.uuid4().hex
    jobs[job_id] = {"status":"running","progress":0,"message":"Starting...","result":None}

    t = threading.Thread(target=run_job, args=(job_id, fpath, ref_paths, max_chunks), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error":"Job not found"}), 404
    return jsonify(job)

@app.route("/report/<filename>")
def download_report(filename):
    path = os.path.join(REPORT_FOLDER, filename)
    if not os.path.exists(path):
        return "Report not found", 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    print("\n  Plagiarism Detector running!")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
