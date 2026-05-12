# =============================
# PDF GENERATOR
# =============================

import os
import re
import datetime
from html import escape

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def clean_latex_formula(text):
    """Convert LaTeX formulas to readable plain text."""
    # Protect currency dollar signs ($300, $50k, etc.) before LaTeX processing
    text = re.sub(r'\$(\d)', r'__CURRENCY__\1', text)
    # Replace common LaTeX math patterns
    text = re.sub(r'\$\\text\{([^}]+)\}\s*/\s*\\text\{([^}]+)\}\$', r'\1 / \2', text)
    text = re.sub(r'\$([^$]+)\$', lambda m: clean_formula_content(m.group(1)), text)
    # Restore currency dollar signs
    text = text.replace('__CURRENCY__', '$')
    return text


def clean_formula_content(formula):
    """Clean LaTeX formula content to readable text."""
    # Remove \text{} wrapper
    formula = re.sub(r'\\text\{([^}]+)\}', r'\1', formula)
    # Remove other LaTeX commands
    formula = re.sub(r'\\[a-zA-Z]+', '', formula)
    # Clean up spacing
    formula = re.sub(r'\s+', ' ', formula).strip()
    return f"({formula})"


def extract_and_convert_tables(text):
    """Extract markdown tables and convert them to readable format."""
    # Pattern to match markdown tables
    table_pattern = r'(\|[^\n]+\|(?:\n\|[-\s:|]+\|)?(?:\n\|[^\n]+\|)*)'
    
    def convert_table(match):
        table_text = match.group(1)
        lines = table_text.strip().split('\n')
        rows = []
        
        for line in lines:
            # Skip separator lines (lines with dashes and colons)
            if re.match(r'^\s*\|\s*[-:\s|]+\s*\|\s*$', line):
                continue
            # Extract cells
            cells = [cell.strip() for cell in line.split('|')]
            cells = [c for c in cells if c]  # Remove empty cells
            if cells:
                rows.append(cells)
        
        if not rows:
            return table_text
        
        # Calculate column widths
        num_cols = max(len(row) for row in rows) if rows else 1
        col_widths = []
        for i in range(num_cols):
            max_width = max(len(str(row[i])) if i < len(row) else 0 for row in rows)
            col_widths.append(max_width)
        
        # Format as text table
        formatted_rows = []
        for row in rows:
            # Pad cells to column width
            padded_cells = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    padded_cells.append(str(cell).ljust(col_widths[i]))
                else:
                    padded_cells.append(str(cell))
            formatted_rows.append(" | ".join(padded_cells))
        
        # Add separator after header
        if len(formatted_rows) > 1:
            separator = "-" * (sum(col_widths) + (len(col_widths) - 1) * 3)
            return formatted_rows[0] + "\n" + separator + "\n" + "\n".join(formatted_rows[1:])
        
        return "\n".join(formatted_rows)
    
    # Replace all tables with formatted versions
    converted = re.sub(table_pattern, convert_table, text)
    return converted


def format_text_for_pdf(text):
    """Convert lightweight markdown content into ReportLab Paragraph markup."""
    content = str(text)
    
    # First, convert markdown tables to readable text format
    content = extract_and_convert_tables(content)
    
    # Clean LaTeX formulas
    content = clean_latex_formula(content)
    
    # Escape HTML
    content = escape(content)
    
    # Convert markdown bold to HTML
    content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
    content = re.sub(r'__(.+?)__', r'<b>\1</b>', content)
    
    # Convert markdown italic to HTML
    content = re.sub(r'\*(.+?)\*', r'<i>\1</i>', content)
    content = re.sub(r'_(.+?)_', r'<i>\1</i>', content)
    
    # Format list items
    content = re.sub(r'(?m)^\s*[-*]\s+', '• ', content)
    
    # Convert markdown Example
    content = re.sub(r'(?m)(^\s*[-*]?\s*)Example:', r'\1<b>Example:</b>', content)
    
    # Convert newlines to line breaks
    content = content.replace("\n", "<br/>")
    
    return content


def format_text_for_streamlit(text):
    """Convert markdown content for Streamlit display."""
    content = str(text)
    # Escape currency dollar signs to prevent Streamlit LaTeX rendering ($300k → \$300k)
    content = re.sub(r'\$(\d)', r'\\$\1', content)
    content = re.sub(r"(?m)(^\s*[-*]?\s*)Example:", r"\1**Example:**", content)
    # Markdown line break behavior requires two trailing spaces before newline.
    return content.replace("\n", "  \n")


def save_report_pdf(tab_name, report_json, time_used, token_usage, output_dir="reports", token_summary=None):
    """Generate and save a PDF report."""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{tab_name}_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    file_path = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph(report_json["title"], styles["Title"]))
    elements.append(Spacer(1, 12))

    for section in report_json["sections"]:
        if "heading" not in section or "content" not in section:
            continue  # Skip invalid sections
        elements.append(Paragraph(section["heading"], styles["Heading2"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(format_text_for_pdf(section["content"]), styles["BodyText"]))
        elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Time used: {time_used:.2f} seconds", styles["BodyText"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Token usage: {token_usage}", styles["BodyText"]))

    # Optional structured token summary (per-call table + totals + performance)
    if token_summary:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Token Usage Details", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        # Per-call table
        per_call = token_summary.get("per_call", [])
        if per_call:
            elements.append(Paragraph("Per LLM Call", styles["Heading3"]))
            elements.append(Spacer(1, 4))
            table_data = [["Step", "Input Tokens", "Output Tokens", "Total Tokens"]]
            for row in per_call:
                table_data.append([
                    str(row.get("step", "")),
                    str(row.get("input_tokens", 0)),
                    str(row.get("output_tokens", 0)),
                    str(row.get("total_tokens", 0)),
                ])
            tbl = Table(table_data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#DCE6F1")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(tbl)
            elements.append(Spacer(1, 8))

        # Totals
        totals = token_summary.get("totals", {})
        if totals:
            elements.append(Paragraph("Totals", styles["Heading3"]))
            elements.append(Spacer(1, 4))
            totals_data = [["Prompt Tokens", "Completion Tokens", "Total Tokens"]]
            totals_data.append([
                str(totals.get("prompt_tokens", 0)),
                str(totals.get("completion_tokens", 0)),
                str(totals.get("total_tokens", 0)),
            ])
            totals_tbl = Table(totals_data, hAlign="LEFT")
            totals_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(totals_tbl)
            elements.append(Spacer(1, 8))

        # Performance
        perf = token_summary.get("performance", {})
        if perf:
            elements.append(Paragraph("Performance", styles["Heading3"]))
            elements.append(Spacer(1, 4))
            perf_data = [["Elapsed (s)", "Completion tok/s", "Total tok/s"]]
            perf_data.append([
                str(perf.get("elapsed_seconds", 0)),
                str(perf.get("completion_tokens_per_sec", 0)),
                str(perf.get("total_tokens_per_sec", 0)),
            ])
            perf_tbl = Table(perf_data, hAlign="LEFT")
            perf_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(perf_tbl)

    doc.build(elements)

    return file_path
