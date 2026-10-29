import numpy as np
from fpdf import FPDF

def detect_anomalies(company_df, threshold=3):
    d = company_df.sort_values('date').copy()
    d['return'] = d['close'].pct_change()
    mean, std = d['return'].mean(), d['return'].std()
    d['z_score'] = (d['return'] - mean) / std
    anomalies = d[abs(d['z_score']) > threshold]
    return anomalies[['date', 'close', 'return', 'z_score']]

def sanitize_for_pdf(text):
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def export_conversation_to_pdf(question, answer, filename="ai_report.pdf"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=14)
        pdf.multi_cell(0, 10, "Beyond the Chart - AI Agent Report")
        pdf.set_font("Helvetica", size=11)
        pdf.ln(5)
        pdf.multi_cell(0, 8, f"Question:\n{sanitize_for_pdf(question)}")
        pdf.ln(3)
        pdf.multi_cell(0, 8, f"Answer:\n{sanitize_for_pdf(answer)}")
        pdf.output(filename)
        return filename
    except Exception as e:
        print(f"PDF export failed: {e}")
        return None