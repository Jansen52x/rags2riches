"""
Multi-Document Image & PDF Generator
Creates visual representations for various document types.
"""
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas
import os
import random
from faker import Faker
from image_generator import get_default_font, wrap_text, COLOR_PALETTES, CHALLENGING_PALETTES
from pdf_generator import PDF_COLOR_SCHEMES, PDF_CHALLENGING_SCHEMES, NumberedCanvas

fake = Faker()


def generate_document_image(document_data, output_path, width=800, height=1000):
    """
    Generates an image for any document type.
    
    Args:
        document_data (dict): Document information including type and content
        output_path (str): Path to save the image
        width (int): Image width
        height (int): Image height
    """
    doc_type = document_data.get('document_type', 'unknown')
    
    if doc_type == 'financial_report':
        generate_financial_report_image(document_data, output_path, width, height)
    elif doc_type == 'press_release':
        generate_press_release_image(document_data, output_path, width, height)
    elif doc_type == 'advertisement':
        generate_advertisement_image(document_data, output_path, width, height)
    elif doc_type == 'partnership_document':
        generate_partnership_image(document_data, output_path, width, height)
    elif doc_type in ['product_brochure', 'services_brochure']:
        generate_brochure_image(document_data, output_path, width, height)
    else:
        generate_generic_document_image(document_data, output_path, width, height)


def generate_financial_report_image(document_data, output_path, width=800, height=1000):
    """Generates a financial report styled image."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Use more formal colors for financial documents
    palette = random.choice([
        {'bg': '#1a237e', 'accent': '#2196f3', 'text': '#ffffff', 'secondary': '#e3f2fd'},
        {'bg': '#263238', 'accent': '#607d8b', 'text': '#ffffff', 'secondary': '#eceff1'},
        {'bg': '#1b5e20', 'accent': '#4caf50', 'text': '#ffffff', 'secondary': '#e8f5e9'},
    ])
    
    # Header
    header_height = height // 5
    draw.rectangle([(0, 0), (width, header_height)], fill=palette['bg'])
    
    # Title
    font_title = get_default_font(36)
    font_subtitle = get_default_font(18)
    font_body = get_default_font(16)
    
    title = document_data.get('title', f'{fake.company()} Financial Report')
    draw.text((60, 40), title, fill=palette['text'], font=font_title)
    
    # Subtitle with quarter/year
    quarter = document_data.get('quarter', '')
    year = document_data.get('year', '')
    draw.text((60, 90), f"{quarter} {year}", fill=palette['accent'], font=font_subtitle)
    
    # Content area
    y_pos = header_height + 60
    margin = 60
    
    # Financial metrics
    revenue = document_data.get('revenue', 0)
    growth = document_data.get('growth_rate', 0)
    profit = document_data.get('profit_margin', 0)
    
    metrics_text = [
        f"Revenue: ${revenue:,.2f}",
        f"Growth Rate: {growth:.1f}%",
        f"Profit Margin: {profit:.1f}%"
    ]
    
    for metric in metrics_text:
        draw.text((margin, y_pos), metric, fill='#333333', font=font_body)
        y_pos += 35
    
    y_pos += 40
    
    # Content
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 2*margin, draw)
    
    for line in wrapped_content[:12]:
        draw.text((margin, y_pos), line, fill='#333333', font=font_body)
        y_pos += 30
    
    img.save(output_path, 'PNG')


def generate_press_release_image(document_data, output_path, width=800, height=1000):
    """Generates a press release styled image."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    palette = random.choice(COLOR_PALETTES)
    
    # Letterhead style header
    draw.rectangle([(0, 0), (width, 100)], fill=palette['bg'])
    
    font_title = get_default_font(32)
    font_subtitle = get_default_font(14)
    font_body = get_default_font(16)
    
    # Company name
    company = document_data.get('company_name', fake.company())
    draw.text((60, 30), company, fill=palette['text'], font=font_title)
    
    # FOR IMMEDIATE RELEASE
    y_pos = 140
    draw.text((60, y_pos), "FOR IMMEDIATE RELEASE", fill=palette['bg'], font=get_default_font(18))
    y_pos += 50
    
    # Title
    title = document_data.get('title', 'Press Release')
    wrapped_title = wrap_text(title, font_title, width - 120, draw)
    for line in wrapped_title[:2]:
        draw.text((60, y_pos), line, fill='#000000', font=get_default_font(24))
        y_pos += 35
    
    y_pos += 30
    
    # Content
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 120, draw)
    
    for line in wrapped_content[:15]:
        draw.text((60, y_pos), line, fill='#333333', font=font_body)
        y_pos += 28
    
    # Contact info at bottom
    contact = document_data.get('contact_email', '')
    draw.rectangle([(60, height - 100), (width - 60, height - 40)], fill=palette['secondary'])
    draw.text((80, height - 80), f"Contact: {contact}", fill='#333333', font=font_subtitle)
    
    img.save(output_path, 'PNG')


def generate_advertisement_image(document_data, output_path, width=600, height=800):
    """Generates an advertisement styled image."""
    img = Image.new('RGB', (width, height), random.choice(['#ff5722', '#2196f3', '#4caf50', '#ff9800', '#9c27b0']))
    draw = ImageDraw.Draw(img)
    
    palette = random.choice(COLOR_PALETTES)
    
    font_title = get_default_font(44)
    font_subtitle = get_default_font(24)
    font_body = get_default_font(18)
    
    # Company name at top
    company = document_data.get('company_name', fake.company())
    bbox = draw.textbbox((0, 0), company, font=font_title)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, 50), company, fill='#ffffff', font=font_title)
    
    # Main content in center
    y_pos = 200
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 80, draw)
    
    for line in wrapped_content[:10]:
        bbox = draw.textbbox((0, 0), line, font=font_body)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y_pos), line, fill='#ffffff', font=font_body)
        y_pos += 32
    
    # Call to action at bottom
    cta = "Contact Us Today!"
    bbox = draw.textbbox((0, 0), cta, font=font_subtitle)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.rectangle([(x - 20, height - 120), (x + text_width + 20, height - 60)], fill='#ffffff')
    draw.text((x, height - 110), cta, fill='#000000', font=font_subtitle)
    
    img.save(output_path, 'PNG')


def generate_partnership_image(document_data, output_path, width=800, height=1000):
    """Generates a partnership document styled image."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    palette = random.choice(COLOR_PALETTES)
    
    font_title = get_default_font(28)
    font_subtitle = get_default_font(20)
    font_body = get_default_font(16)
    
    # Header with both company names
    companies = document_data.get('companies', [])
    if len(companies) >= 2:
        # Company 1 side
        draw.rectangle([(0, 0), (width // 2, 120)], fill=palette['bg'])
        draw.text((30, 50), companies[0], fill=palette['text'], font=font_subtitle)
        
        # Company 2 side
        draw.rectangle([(width // 2, 0), (width, 120)], fill=palette['accent'])
        draw.text((width // 2 + 30, 50), companies[1], fill='#000000', font=font_subtitle)
    
    # Partnership symbol in middle
    draw.text((width // 2 - 15, 140), "⟷", fill=palette['bg'], font=get_default_font(48))
    
    # Title
    y_pos = 200
    title = document_data.get('title', 'Partnership Agreement')
    wrapped_title = wrap_text(title, font_title, width - 120, draw)
    for line in wrapped_title[:2]:
        draw.text((60, y_pos), line, fill='#000000', font=font_title)
        y_pos += 35
    
    y_pos += 40
    
    # Content
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 120, draw)
    
    for line in wrapped_content[:15]:
        draw.text((60, y_pos), line, fill='#333333', font=font_body)
        y_pos += 28
    
    img.save(output_path, 'PNG')


def generate_brochure_image(document_data, output_path, width=800, height=1000):
    """Generates a generic brochure image."""
    from image_generator import generate_company_brochure
    # Reuse existing brochure generator but with document content
    generate_company_brochure(document_data, output_path, width, height)


def generate_generic_document_image(document_data, output_path, width=800, height=1000):
    """Generates a generic document styled image."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    palette = random.choice(COLOR_PALETTES)
    
    font_title = get_default_font(32)
    font_body = get_default_font(16)
    
    # Simple header
    draw.rectangle([(0, 0), (width, 80)], fill=palette['bg'])
    
    company = document_data.get('company_name', 'Company')
    draw.text((60, 25), company, fill=palette['text'], font=font_title)
    
    # Title
    y_pos = 120
    title = document_data.get('title', 'Document')
    draw.text((60, y_pos), title, fill='#000000', font=get_default_font(24))
    y_pos += 60
    
    # Content
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 120, draw)
    
    for line in wrapped_content[:18]:
        draw.text((60, y_pos), line, fill='#333333', font=font_body)
        y_pos += 28
    
    img.save(output_path, 'PNG')


def generate_document_pdf(document_data, output_path):
    """
    Generates a PDF for any document type.
    
    Args:
        document_data (dict): Document information including type and content
        output_path (str): Path to save the PDF
    """
    doc_type = document_data.get('document_type', 'unknown')
    
    if doc_type == 'financial_report':
        generate_financial_report_pdf(document_data, output_path)
    elif doc_type in ['product_brochure', 'services_brochure']:
        generate_brochure_pdf(document_data, output_path)
    else:
        generate_generic_document_pdf(document_data, output_path)


def generate_financial_report_pdf(document_data, output_path):
    """Generates a financial report PDF."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=20
    )
    story.append(Paragraph(document_data.get('title', 'Financial Report'), title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Financial data table
    metrics = [
        ['Metric', 'Value'],
        ['Revenue', f"${document_data.get('revenue', 0):,.2f}"],
        ['Growth Rate', f"{document_data.get('growth_rate', 0):.1f}%"],
        ['Profit Margin', f"{document_data.get('profit_margin', 0):.1f}%"]
    ]
    
    table = Table(metrics, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.4*inch))
    
    # Content
    content = document_data.get('content', '')
    story.append(Paragraph(content, styles['BodyText']))
    
    doc.build(story)


def generate_brochure_pdf(document_data, output_path):
    """Generates a brochure PDF."""
    from pdf_generator import generate_pdf_brochure
    # Reuse existing PDF generator
    generate_pdf_brochure(document_data, output_path)


def generate_generic_document_pdf(document_data, output_path):
    """Generates a generic document PDF."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    story.append(Paragraph(document_data.get('title', fake.catch_phrase()), styles['Title']))
    story.append(Spacer(1, 0.3*inch))
    
    # Company name
    company = document_data.get('company_name', fake.company())
    story.append(Paragraph(f"<b>{company}</b>", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Content
    content = document_data.get('content', '')
    story.append(Paragraph(content, styles['BodyText']))
    
    doc.build(story)
