"""
Script to generate standalone PDF files for all 30 Research Papers
and save them into the 'papers_pdf/' directory.
"""

import os
from fpdf import FPDF
from research_data import RESEARCH_PAPERS

def clean_text(t):
    if not isinstance(t, str):
        t = str(t)
    cleaned = t.replace('’', "'").replace('“', '"').replace('”', '"').replace('–', '-').replace('—', '-')
    for char in ['🎓', '📚', '💳', '⚠️', '💸', '🧠', '📉', '✏️', '🔍', '✅', '📈', '💰', '🏠', '🔴', '🟡', '🟢', '🟣', '⚡', '🧘', '⏰', '🚨', '📖', '🎯', '🛠️', '📄']:
        cleaned = cleaned.replace(char, '')
    return cleaned.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_for_paper(paper, output_dir):
    pdf = FPDF()
    pdf.add_page()
    
    # Title Header Banner
    pdf.set_fill_color(24, 20, 56)
    pdf.rect(0, 0, 210, 36, "F")
    
    pdf.set_y(8)
    pdf.set_text_color(240, 147, 251)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 6, clean_text(f"PAPER #{paper['id']}: RESEARCH BRIEF & METADATA"), 0, 1, "C")
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 5, clean_text("EduGuard Early Warning & Student Retention Research Foundation"), 0, 1, "C")
    
    pdf.set_y(42)
    pdf.set_text_color(48, 43, 99)
    pdf.set_font("Arial", "B", 13)
    pdf.multi_cell(0, 6, clean_text(paper['title']))
    
    pdf.ln(2)
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(25, 5, "Authors:", 0, 0, "L")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 5, clean_text(paper['authors']), 0, 1, "L")
    
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(25, 5, "Publication:", 0, 0, "L")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 5, clean_text(f"{paper['journal']} ({paper['year']})"), 0, 1, "L")
    
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(25, 5, "Dataset:", 0, 0, "L")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 5, clean_text(paper['dataset']), 0, 1, "L")
    
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(25, 5, "Algorithms:", 0, 0, "L")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 5, clean_text(", ".join(paper['algorithms'])), 0, 1, "L")
    
    pdf.ln(3)
    
    # Abstract Section
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, pdf.get_y(), 190, 6, "F")
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "  Full Abstract", 0, 1, "L")
    pdf.ln(1)
    
    pdf.set_font("Arial", "", 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, clean_text(paper['abstract']))
    pdf.ln(3)
    
    # Key Findings
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, pdf.get_y(), 190, 6, "F")
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "  Key Findings & Methodology", 0, 1, "L")
    pdf.ln(1)
    
    pdf.set_font("Arial", "", 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, clean_text(paper['key_findings']))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 8.5)
    pdf.cell(45, 5, "Top Performing Model:", 0, 0, "L")
    pdf.set_font("Arial", "", 8.5)
    pdf.cell(0, 5, clean_text(paper['best_algorithm']), 0, 1, "L")
    pdf.ln(3)
    
    # EduGuard Integration Blueprint
    pdf.set_fill_color(237, 233, 254)
    pdf.rect(10, pdf.get_y(), 190, 6, "F")
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(88, 28, 135)
    pdf.cell(0, 6, "  EduGuard System Integration Blueprint", 0, 1, "L")
    pdf.ln(1)
    
    pdf.set_font("Arial", "", 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, clean_text(paper['eduguard_integration']))
    pdf.ln(2)
    
    if "metrics" in paper:
        pdf.set_font("Arial", "B", 8.5)
        pdf.cell(0, 5, "Reported Benchmark Metrics:", 0, 1, "L")
        pdf.set_font("Arial", "", 8.5)
        for k, v in paper["metrics"].items():
            pdf.cell(0, 4.5, clean_text(f"- {k}: {v}"), 0, 1, "L")
            
    # Footer
    pdf.set_y(280)
    pdf.set_font("Arial", "I", 7.5)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 4, clean_text("EduGuard Academic Repository | Synthesized from 30 Literature Publications"), 0, 0, "C")
    
    filename = f"Paper_{paper['id']:02d}_{paper['best_algorithm'].replace(' ', '_').replace('/', '_')}.pdf"
    filepath = os.path.join(output_dir, filename)
    pdf.output(filepath)
    return filepath

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "papers_pdf")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating 30 paper PDFs in: {output_dir}")
    generated = []
    for paper in RESEARCH_PAPERS:
        path = generate_pdf_for_paper(paper, output_dir)
        generated.append(path)
        print(f"Generated Paper #{paper['id']}: {os.path.basename(path)}")
        
    print(f"\nSuccessfully generated {len(generated)} PDF files for all 30 papers!")

if __name__ == "__main__":
    main()
