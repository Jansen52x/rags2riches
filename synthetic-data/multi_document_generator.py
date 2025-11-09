"""
Multi-Document Image & PDF Generator
Creates visual representations for various document types.
"""
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
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
    """Generates a visually striking financial report styled image."""
    img = Image.new('RGB', (width, height), '#F5F5F5')  # Light gray background
    draw = ImageDraw.Draw(img)
    
    # Use professional financial colors
    palette = random.choice([
        {'bg': '#1565C0', 'accent': '#4CAF50', 'text': '#ffffff', 'secondary': '#E3F2FD', 'chart': '#FFC107'},
        {'bg': '#0D47A1', 'accent': '#00C853', 'text': '#ffffff', 'secondary': '#E8F5E9', 'chart': '#FF9800'},
        {'bg': '#004D40', 'accent': '#00BFA5', 'text': '#ffffff', 'secondary': '#E0F2F1', 'chart': '#FFD600'},
    ])
    
    # HEADER with modern design
    header_height = 180
    draw.rectangle([(0, 0), (width, header_height)], fill=palette['bg'])
    
    # Accent stripe at top
    draw.rectangle([(0, 0), (width, 12)], fill=palette['accent'])
    
    # Title fonts
    font_title = get_default_font(42)
    font_subtitle = get_default_font(22)
    font_body = get_default_font(18)
    font_metrics = get_default_font(24)
    font_small = get_default_font(14)
    
    title = document_data.get('title', 'Financial Report')
    draw.text((50, 30), title, fill=palette['text'], font=font_title)
    
    # Quarter/Year with styled background
    quarter = document_data.get('quarter', '')
    year = document_data.get('year', '')
    period_text = f"{quarter} {year}"
    bbox = draw.textbbox((0, 0), period_text, font=font_subtitle)
    text_width = bbox[2] - bbox[0]
    draw.rectangle([(50, 90), (70 + text_width, 125)], fill=palette['accent'])
    draw.text((60, 95), period_text, fill='#000000', font=font_subtitle)
    
    # Decorative corner element
    draw.ellipse([(width - 100, header_height - 80), (width - 20, header_height)], 
                fill=palette['accent'], outline=palette['text'], width=3)
    
    # METRICS SECTION with cards
    metrics_y = header_height + 30
    margin = 40
    
    # Financial metrics in styled cards
    revenue = document_data.get('revenue', 0)
    growth = document_data.get('growth_rate', 0)
    profit = document_data.get('profit_margin', 0)
    
    card_width = (width - 3 * margin) // 3
    card_height = 110
    
    metrics = [
        ('Revenue', f'${revenue:,.0f}', palette['accent']),
        ('Growth', f'{growth:.1f}%', palette['chart']),
        ('Profit Margin', f'{profit:.1f}%', palette['accent'])
    ]
    
    for idx, (label, value, color) in enumerate(metrics):
        card_x = margin + idx * (card_width + margin // 2)
        
        # Card with shadow
        draw.rectangle([(card_x + 5, metrics_y + 5), 
                       (card_x + card_width + 5, metrics_y + card_height + 5)], 
                       fill='#00000020')  # Shadow
        draw.rectangle([(card_x, metrics_y), 
                       (card_x + card_width, metrics_y + card_height)], 
                       fill='#FFFFFF', outline=color, width=3)
        
        # Accent bar in card
        draw.rectangle([(card_x, metrics_y), (card_x + card_width, metrics_y + 8)], fill=color)
        
        # Label
        draw.text((card_x + 15, metrics_y + 20), label, fill='#666666', font=font_small)
        
        # Value (large and bold)
        value_bbox = draw.textbbox((0, 0), value, font=font_metrics)
        value_width = value_bbox[2] - value_bbox[0]
        value_x = card_x + (card_width - value_width) // 2
        draw.text((value_x, metrics_y + 50), value, fill='#000000', font=font_metrics)
    
    # CONTENT SECTION
    content_y = metrics_y + card_height + 40
    
    # Section header
    draw.rectangle([(margin, content_y), (width - margin, content_y + 45)], 
                  fill=palette['secondary'], outline=palette['bg'], width=2)
    draw.text((margin + 20, content_y + 10), "Executive Summary", 
             fill=palette['bg'], font=font_subtitle)
    
    content_y += 65
    
    # Content with left accent bar
    draw.rectangle([(margin, content_y), (margin + 6, height - 100)], fill=palette['accent'])
    
    # Content text
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 2*margin - 30, draw)
    
    y_pos = content_y + 10
    for line in wrapped_content[:14]:
        if y_pos > height - 120:
            break
        draw.text((margin + 25, y_pos), line, fill='#2C2C2C', font=font_body)
        y_pos += 28
    
    # FOOTER
    footer_y = height - 70
    draw.rectangle([(0, footer_y), (width, height)], fill=palette['bg'])
    draw.rectangle([(0, footer_y), (width, footer_y + 4)], fill=palette['accent'])
    
    footer_text = f"Confidential | {document_data.get('company_name', 'Company')} Financial Report"
    draw.text((margin, footer_y + 25), footer_text, fill=palette['text'], font=font_small)
    
    img.save(output_path, 'PNG')


def generate_press_release_image(document_data, output_path, width=800, height=1000):
    """Generates a modern, eye-catching press release styled image."""
    img = Image.new('RGB', (width, height), '#FAFAFA')
    draw = ImageDraw.Draw(img)
    
    palette = random.choice([
        {'bg': '#D32F2F', 'accent': '#FFC107', 'text': '#ffffff', 'secondary': '#FFEBEE'},
        {'bg': '#1976D2', 'accent': '#FF9800', 'text': '#ffffff', 'secondary': '#E3F2FD'},
        {'bg': '#7B1FA2', 'accent': '#00E676', 'text': '#ffffff', 'secondary': '#F3E5F5'},
        {'bg': '#0288D1', 'accent': '#FFEB3B', 'text': '#ffffff', 'secondary': '#E1F5FE'},
    ])
    
    # MODERN HEADER with angled design
    draw.rectangle([(0, 0), (width, 140)], fill=palette['bg'])
    
    # Angled accent stripe
    points = [(0, 110), (width, 140), (width, 110), (0, 80)]
    draw.polygon(points, fill=palette['accent'])
    
    font_title = get_default_font(38)
    font_subtitle = get_default_font(16)
    font_body = get_default_font(18)
    font_header = get_default_font(28)
    font_small = get_default_font(13)
    
    # Company name with icon
    company = document_data.get('company_name', 'Company')
    draw.text((60, 30), company, fill=palette['text'], font=font_title)
    
    # Decorative circle icon
    draw.ellipse([(width - 100, 20), (width - 30, 90)], fill=palette['accent'], outline=palette['text'], width=3)
    draw.text((width - 75, 42), "PR", fill=palette['bg'], font=font_header)
    
    # FOR IMMEDIATE RELEASE badge
    y_pos = 160
    badge_text = "FOR IMMEDIATE RELEASE"
    bbox = draw.textbbox((0, 0), badge_text, font=font_subtitle)
    badge_width = bbox[2] - bbox[0]
    
    draw.rectangle([(60, y_pos), (80 + badge_width, y_pos + 30)], 
                  fill=palette['bg'], outline=palette['accent'], width=2)
    draw.text((70, y_pos + 5), badge_text, fill=palette['text'], font=font_subtitle)
    
    y_pos += 55
    
    # Title with background highlight
    title = document_data.get('title', 'Press Release')
    wrapped_title = wrap_text(title, font_header, width - 120, draw)
    
    for idx, line in enumerate(wrapped_title[:2]):
        # Highlight background
        bbox = draw.textbbox((0, 0), line, font=font_header)
        line_width = bbox[2] - bbox[0]
        draw.rectangle([(55, y_pos - 5), (75 + line_width, y_pos + 35)], 
                      fill=palette['secondary'])
        draw.text((60, y_pos), line, fill=palette['bg'], font=font_header)
        y_pos += 40
    
    y_pos += 20
    
    # Decorative separator line
    draw.line([(60, y_pos), (width - 60, y_pos)], fill=palette['accent'], width=4)
    draw.line([(60, y_pos + 6), (width - 60, y_pos + 6)], fill=palette['bg'], width=2)
    
    y_pos += 30
    
    # CONTENT with side decoration
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 160, draw)
    
    # Vertical accent bar
    draw.rectangle([(50, y_pos), (58, height - 130)], fill=palette['accent'])
    
    # Content with better spacing
    for idx, line in enumerate(wrapped_content[:16]):
        if y_pos > height - 150:
            break
        
        # Alternate subtle backgrounds for readability
        if idx % 5 == 0 and idx > 0:
            draw.rectangle([(70, y_pos - 3), (width - 60, y_pos + 25)], fill=palette['secondary'])
        
        draw.text((75, y_pos), line, fill='#1A1A1A', font=font_body)
        y_pos += 26
    
    # CONTACT FOOTER with modern design
    contact_y = height - 110
    draw.rectangle([(0, contact_y), (width, height)], fill=palette['bg'])
    
    # Decorative angle
    points = [(0, contact_y), (width, contact_y), (width, contact_y + 30), (0, contact_y + 10)]
    draw.polygon(points, fill=palette['accent'])
    
    contact = document_data.get('contact_email', '')
    if contact:
        draw.text((60, contact_y + 45), "MEDIA CONTACT", fill=palette['accent'], font=font_small)
        draw.text((60, contact_y + 65), contact, fill=palette['text'], font=font_subtitle)
    
    # Decorative corner accent
    draw.rectangle([(width - 60, height - 60), (width, height)], fill=palette['accent'])
    
    img.save(output_path, 'PNG')


def generate_advertisement_image(document_data, output_path, width=600, height=800):
    """Generates a vibrant, attention-grabbing advertisement styled image."""
    # Start with bold gradient background
    base_colors = ['#FF5722', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63']
    bg_color = random.choice(base_colors)
    
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Create radial gradient effect (simulated)
    center_x, center_y = width // 2, height // 3
    for radius in range(400, 0, -10):
        alpha = int(50 * (radius / 400))
        color_value = max(0, 255 - alpha)
        circle_color = f'#{color_value:02x}{color_value:02x}{color_value:02x}'
        draw.ellipse([(center_x - radius, center_y - radius), 
                     (center_x + radius, center_y + radius)], 
                     fill=circle_color + '20')
    
    palette = random.choice(COLOR_PALETTES)
    
    font_title = get_default_font(52)
    font_subtitle = get_default_font(28)
    font_body = get_default_font(20)
    font_cta = get_default_font(32)
    
    # DYNAMIC TOP SECTION
    # Angled accent stripe
    points = [(0, 0), (width, 40), (width, 0)]
    draw.polygon(points, fill='#FFFFFF40')
    
    # Company name with dynamic styling
    company = document_data.get('company_name', 'Company')
    y_pos = 60
    
    # Company name with outline effect
    for offset in [(2,2), (-2,-2), (2,-2), (-2,2)]:
        draw.text((width//2 - 150 + offset[0], y_pos + offset[1]), company, 
                 fill='#00000080', font=font_title)
    draw.text((width//2 - 150, y_pos), company, fill='#FFFFFF', font=font_title)
    
    # Decorative stars/sparkles
    star_positions = [(50, 50), (width-50, 80), (100, 120), (width-100, 140)]
    for sx, sy in star_positions:
        draw.text((sx, sy), "★", fill='#FFFFFF', font=get_default_font(24))
    
    y_pos = 200
    
    # CONTENT in styled boxes
    content = document_data.get('content', '')
    wrapped_content = wrap_text(content, font_body, width - 100, draw)
    
    # Content background card
    card_top = y_pos - 20
    card_bottom = min(y_pos + len(wrapped_content[:8]) * 32 + 40, height - 200)
    
    # Card shadow
    draw.rectangle([(55, card_top + 5), (width - 45, card_bottom + 5)], fill='#00000040')
    # Card
    draw.rectangle([(50, card_top), (width - 50, card_bottom)], 
                  fill='#FFFFFF', outline='#000000', width=3)
    
    # Accent corner on card
    draw.rectangle([(50, card_top), (70, card_top + 60)], fill=bg_color)
    draw.rectangle([(width - 70, card_top), (width - 50, card_top + 60)], fill=bg_color)
    
    # Content text
    for idx, line in enumerate(wrapped_content[:8]):
        if y_pos > card_bottom - 40:
            break
        
        # Center align text
        bbox = draw.textbbox((0, 0), line, font=font_body)
        line_width = bbox[2] - bbox[0]
        x = (width - line_width) // 2
        draw.text((x, y_pos), line, fill='#1A1A1A', font=font_body)
        y_pos += 32
    
    # CALL TO ACTION BUTTON
    cta_y = height - 150
    cta_text = "Learn More →"
    bbox = draw.textbbox((0, 0), cta_text, font=font_cta)
    cta_width = bbox[2] - bbox[0]
    
    button_padding = 40
    button_x1 = (width - cta_width - 2*button_padding) // 2
    button_y1 = cta_y
    button_x2 = button_x1 + cta_width + 2*button_padding
    button_y2 = cta_y + 70
    
    # Button shadow
    draw.rectangle([(button_x1 + 5, button_y1 + 5), (button_x2 + 5, button_y2 + 5)], 
                  fill='#00000050')
    
    # Button with gradient effect
    for i in range(5):
        offset = i * 2
        draw.rectangle([(button_x1 - offset, button_y1 - offset), 
                       (button_x2 + offset, button_y2 + offset)], 
                       outline='#FFFFFF', width=1)
    
    draw.rectangle([(button_x1, button_y1), (button_x2, button_y2)], 
                  fill='#FFFFFF', outline='#000000', width=4)
    
    # CTA text
    text_x = (width - cta_width) // 2
    draw.text((text_x, cta_y + 15), cta_text, fill='#000000', font=font_cta)
    
    # Decorative elements
    # Circles in corners
    draw.ellipse([(10, height - 80), (60, height - 30)], fill='#FFFFFF40')
    draw.ellipse([(width - 60, height - 80), (width - 10, height - 30)], fill='#FFFFFF40')
    
    # Dynamic lines
    for i in range(3):
        y = height - 25 + i * 8
        draw.line([(width//4, y), (3*width//4, y)], fill='#FFFFFF60', width=2)
    
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
    """Generates a visually appealing brochure image using the actual Gemini-generated content."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Use vibrant, modern color schemes
    palette = random.choice([
        {'bg': '#0078D4', 'accent': '#FFB900', 'text': '#ffffff', 'secondary': '#E3F2FD', 'dark': '#00188F'},  # Microsoft Blue
        {'bg': '#1565C0', 'accent': '#FFC107', 'text': '#ffffff', 'secondary': '#E8F5E9', 'dark': '#0D47A1'},  # Professional Blue
        {'bg': '#00897B', 'accent': '#FF6F00', 'text': '#ffffff', 'secondary': '#E0F2F1', 'dark': '#004D40'},  # Teal
        {'bg': '#6A1B9A', 'accent': '#FDD835', 'text': '#ffffff', 'secondary': '#F3E5F5', 'dark': '#4A148C'},  # Purple
        {'bg': '#D32F2F', 'accent': '#FFC107', 'text': '#ffffff', 'secondary': '#FFEBEE', 'dark': '#B71C1C'},  # Red
    ])
    
    # HEADER WITH GRADIENT EFFECT (simulated with rectangles)
    header_height = height // 4
    # Create gradient effect with overlapping rectangles
    gradient_steps = 20
    for i in range(gradient_steps):
        alpha = int(255 * (1 - i / gradient_steps))
        y = int(i * header_height / gradient_steps)
        h = int(header_height / gradient_steps) + 1
        draw.rectangle([(0, y), (width, y + h)], fill=palette['bg'])
    
    # Add decorative accent bar at top
    draw.rectangle([(0, 0), (width, 15)], fill=palette['accent'])
    
    # Add decorative geometric shapes in header
    # Circle in top right
    circle_size = 120
    draw.ellipse([(width - circle_size - 40, 30), (width - 40, 30 + circle_size)], 
                 fill=palette['dark'], outline=palette['accent'], width=3)
    
    # Title section
    font_title = get_default_font(48)
    font_subtitle = get_default_font(24)
    font_body = get_default_font(18)
    font_small = get_default_font(14)
    
    company = document_data.get('company_name', 'Company')
    doc_type = document_data.get('document_type', 'brochure').replace('_', ' ').title()
    
    # Company name with shadow effect
    shadow_offset = 3
    draw.text((63, 63), company, fill='#00000040', font=font_title)  # Shadow
    draw.text((60, 60), company, fill=palette['text'], font=font_title)  # Main text
    
    # Document type with background pill
    bbox = draw.textbbox((0, 0), doc_type, font=font_subtitle)
    text_width = bbox[2] - bbox[0]
    pill_padding = 20
    pill_y = 130
    draw.rectangle([(60 - pill_padding, pill_y - 5), 
                   (60 + text_width + pill_padding, pill_y + 30)], 
                   fill=palette['accent'], outline=palette['dark'], width=2)
    draw.text((60, pill_y), doc_type, fill=palette['dark'], font=font_subtitle)
    
    # Decorative diagonal lines in header
    for i in range(5):
        x_start = width - 200 + i * 30
        draw.line([(x_start, header_height - 80), (x_start + 60, header_height - 20)], 
                 fill=palette['accent'], width=2)
    
    # Separator with gradient
    separator_y = header_height
    draw.rectangle([(0, separator_y), (width, separator_y + 8)], fill=palette['accent'])
    draw.rectangle([(0, separator_y + 8), (width, separator_y + 10)], fill=palette['dark'])
    
    # CONTENT SECTION with styled background
    content_start_y = separator_y + 40
    margin = 50
    
    # Add subtle background pattern
    for i in range(5):
        y_offset = content_start_y + i * 180
        draw.rectangle([(margin - 10, y_offset), (margin - 5, y_offset + 100)], 
                      fill=palette['accent'])
    
    # Content title bar
    draw.rectangle([(margin, content_start_y), (width - margin, content_start_y + 40)], 
                  fill=palette['secondary'], outline=palette['bg'], width=2)
    draw.text((margin + 15, content_start_y + 8), "Overview", fill=palette['dark'], font=font_subtitle)
    
    # Get the IMAGE-specific content (variation A)
    content = document_data.get('content_image', document_data.get('content', ''))
    
    # Content area with better formatting
    y_pos = content_start_y + 60
    line_height = 26
    
    # Wrap and display the content with better line spacing
    wrapped_content = wrap_text(content, font_body, width - 2*margin - 20, draw)
    
    # Display content lines with alternating subtle backgrounds
    for idx, line in enumerate(wrapped_content[:18]):
        if y_pos + line_height > height - 100:
            break
        
        # Add subtle background for better readability every few lines
        if idx % 4 == 0 and idx > 0:
            draw.rectangle([(margin - 5, y_pos - 5), (width - margin + 5, y_pos + line_height - 5)], 
                          fill=palette['secondary'] + '40')  # Semi-transparent
        
        draw.text((margin + 10, y_pos), line, fill='#1A1A1A', font=font_body)
        y_pos += line_height
    
    # FOOTER with contact info and branding
    footer_y = height - 80
    draw.rectangle([(0, footer_y), (width, height)], fill=palette['bg'])
    draw.rectangle([(0, footer_y), (width, footer_y + 5)], fill=palette['accent'])
    
    # Footer text
    footer_text = f"© {company} | Professional Solutions"
    draw.text((margin, footer_y + 25), footer_text, fill=palette['text'], font=font_small)
    
    # Decorative corner elements
    corner_size = 30
    draw.rectangle([(width - corner_size - 20, footer_y + 15), 
                   (width - 20, footer_y + 15 + corner_size)], 
                   fill=palette['accent'])
    
    img.save(output_path, 'PNG')


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
    """Generates a brochure PDF using the actual Gemini-generated content."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    # Company name
    company = document_data.get('company_name', 'Company')
    story.append(Paragraph(company, title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Document type subtitle
    doc_type = document_data.get('document_type', 'brochure').replace('_', ' ').title()
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#2196f3'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph(doc_type, subtitle_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Horizontal line
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#2196f3'), 
                           spaceAfter=0.3*inch, spaceBefore=0.1*inch))
    
    # Content - USE THE PDF-SPECIFIC CONTENT (variation B)!
    content = document_data.get('content_pdf', document_data.get('content', ''))
    
    # Create justified body text style
    body_style = ParagraphStyle(
        'BodyJustified',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        leading=16
    )
    
    story.append(Paragraph(content, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Contact information if available
    contact_email = document_data.get('contact_email')
    if contact_email:
        story.append(Spacer(1, 0.2*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        contact_style = ParagraphStyle(
            'Contact',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=10
        )
        story.append(Paragraph(f"<b>Contact:</b> {contact_email}", contact_style))
    
    doc.build(story)


def generate_generic_document_pdf(document_data, output_path):
    """Generates a generic document PDF."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    story.append(Paragraph(document_data.get('title', 'Document'), styles['Title']))
    story.append(Spacer(1, 0.3*inch))
    
    # Company name
    company = document_data.get('company_name', 'Company')
    story.append(Paragraph(f"<b>{company}</b>", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Content
    content = document_data.get('content', '')
    story.append(Paragraph(content, styles['BodyText']))
    
    doc.build(story)
