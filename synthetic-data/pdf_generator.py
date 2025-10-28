"""
PDF Generator Module
Generates professional PDF brochures using ReportLab library.
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import os
import random


# Color schemes for PDF brochures
PDF_COLOR_SCHEMES = [
    {
        'primary': colors.HexColor('#1a237e'),
        'secondary': colors.HexColor('#00bcd4'),
        'accent': colors.HexColor('#e3f2fd'),
        'text': colors.black
    },
    {
        'primary': colors.HexColor('#004d40'),
        'secondary': colors.HexColor('#00bfa5'),
        'accent': colors.HexColor('#e0f2f1'),
        'text': colors.black
    },
    {
        'primary': colors.HexColor('#311b92'),
        'secondary': colors.HexColor('#7c4dff'),
        'accent': colors.HexColor('#ede7f6'),
        'text': colors.black
    },
    {
        'primary': colors.HexColor('#b71c1c'),
        'secondary': colors.HexColor('#ff5252'),
        'accent': colors.HexColor('#ffebee'),
        'text': colors.black
    },
    {
        'primary': colors.HexColor('#e65100'),
        'secondary': colors.HexColor('#ff9800'),
        'accent': colors.HexColor('#fff3e0'),
        'text': colors.black
    },
    {
        'primary': colors.HexColor('#01579b'),
        'secondary': colors.HexColor('#0288d1'),
        'accent': colors.HexColor('#e1f5fe'),
        'text': colors.black
    },
    {
        'primary': colors.HexColor('#33691e'),
        'secondary': colors.HexColor('#689f38'),
        'accent': colors.HexColor('#f1f8e9'),
        'text': colors.black
    },
]

# Challenging color schemes for PDFs (poor contrast)
PDF_CHALLENGING_SCHEMES = [
    {
        'primary': colors.HexColor('#2c2c2c'),
        'secondary': colors.HexColor('#404040'),
        'accent': colors.HexColor('#3a3a3a'),
        'text': colors.HexColor('#444444')
    },
    {
        'primary': colors.HexColor('#e0e0e0'),
        'secondary': colors.HexColor('#f5f5f5'),
        'accent': colors.HexColor('#fafafa'),
        'text': colors.HexColor('#eeeeee')
    },
    {
        'primary': colors.HexColor('#1976d2'),
        'secondary': colors.HexColor('#2196f3'),
        'accent': colors.HexColor('#42a5f5'),
        'text': colors.HexColor('#64b5f6')
    },
]

# Layout variations for PDFs
PDF_LAYOUTS = ['standard', 'modern', 'minimal', 'bold']


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for adding page numbers and headers/footers."""
    
    def __init__(self, *args, **kwargs):
        self.company_name = kwargs.pop('company_name', 'Company')
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)
        page_num = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 0.75*inch, 0.5*inch, page_num)
        self.drawString(0.75*inch, 0.5*inch, self.company_name)


def create_custom_styles(color_scheme, layout_style='standard'):
    """
    Creates custom paragraph styles with the given color scheme and layout.
    
    Args:
        color_scheme (dict): Dictionary containing color definitions
        layout_style (str): Layout style to apply
        
    Returns:
        dict: Dictionary of custom styles
    """
    styles = getSampleStyleSheet()
    
    # Vary font sizes based on layout
    if layout_style == 'bold':
        title_size, subtitle_size, heading_size, body_size = 32, 18, 16, 12
    elif layout_style == 'minimal':
        title_size, subtitle_size, heading_size, body_size = 24, 14, 12, 10
    else:
        title_size, subtitle_size, heading_size, body_size = 28, 16, 14, 11
    
    # Title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=title_size,
        textColor=color_scheme['primary'],
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    # Subtitle style
    styles.add(ParagraphStyle(
        name='CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=subtitle_size,
        textColor=color_scheme['secondary'],
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    # Section heading style
    styles.add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontSize=heading_size,
        textColor=color_scheme['primary'],
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    ))
    
    # Body text style - vary text color based on layout
    body_color = colors.HexColor('#333333')
    if layout_style == 'bold':
        body_color = color_scheme['text']
    
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['BodyText'],
        fontSize=body_size,
        textColor=body_color,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=body_size + 5
    ))
    
    # Contact info style
    styles.add(ParagraphStyle(
        name='ContactInfo',
        parent=styles['BodyText'],
        fontSize=body_size,
        textColor=body_color,
        spaceAfter=8,
        leading=body_size + 3
    ))
    
    return styles


def generate_pdf_brochure(client_data, output_path):
    """
    Generates a professional multi-page PDF brochure with varied layouts for a company.
    
    Args:
        client_data (dict): Dictionary containing company information
        output_path (str): Path to save the PDF file
    """
    # Extract data
    company_name = client_data.get('company_name', 'Company Name')
    industry = client_data.get('industry', 'Industry')
    contact_person = client_data.get('contact_person', 'Contact Person')
    contact_email = client_data.get('contact_email', 'email@example.com')
    description = client_data.get('company_description', 'Company description.')
    
    # 20% chance to use challenging color scheme
    if random.random() < 0.2:
        color_scheme = random.choice(PDF_CHALLENGING_SCHEMES)
    else:
        color_scheme = random.choice(PDF_COLOR_SCHEMES)
    
    # Select random layout style
    layout_style = random.choice(PDF_LAYOUTS)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=1*inch
    )
    
    # Get custom styles with layout variation
    styles = create_custom_styles(color_scheme, layout_style)
    
    # Container for the 'Flowable' objects
    story = []
    
    # ===== PAGE 1: COVER PAGE =====
    
    # Add spacer for top margin - vary spacing
    story.append(Spacer(1, random.choice([1.2, 1.5, 1.8])*inch))
    
    # Company name (title)
    story.append(Paragraph(company_name, styles['CustomTitle']))
    story.append(Spacer(1, random.choice([0.2, 0.3, 0.4])*inch))
    
    # Industry (subtitle)
    story.append(Paragraph(industry.title(), styles['CustomSubtitle']))
    story.append(Spacer(1, random.choice([0.3, 0.5, 0.7])*inch))
    
    # Decorative horizontal line - vary appearance
    if random.random() > 0.2:  # 80% chance to show line
        story.append(HRFlowable(
            width=random.choice(["60%", "80%", "100%"]),
            thickness=random.choice([1, 2, 3]),
            color=color_scheme['secondary'],
            spaceBefore=10,
            spaceAfter=10,
            hAlign='CENTER'
        ))
    
    story.append(Spacer(1, random.choice([0.3, 0.5, 0.7])*inch))
    
    # Tagline/description snippet
    tagline = description.split('.')[0] + '.'
    tagline_style = ParagraphStyle(
        name='Tagline',
        parent=styles['CustomBody'],
        fontSize=random.choice([11, 13, 15]),
        alignment=TA_CENTER,
        textColor=colors.HexColor('#555555') if random.random() > 0.15 else color_scheme['accent'],
        italic=True
    )
    story.append(Paragraph(tagline, tagline_style))
    
    story.append(Spacer(1, random.choice([1.5, 2, 2.5])*inch))
    
    # Contact box on cover - vary style
    contact_data = [
        ['Contact Information'],
        [f'<b>Email:</b> {contact_email}'],
        [f'<b>Contact:</b> {contact_person}'],
    ]
    
    contact_table = Table(contact_data, colWidths=[4.5*inch])
    
    # Vary table styling
    header_bg = color_scheme['primary'] if random.random() > 0.2 else color_scheme['secondary']
    body_bg = color_scheme['accent'] if random.random() > 0.15 else colors.white
    
    contact_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), random.choice([11, 12, 13])),
        ('FONTSIZE', (0, 1), (-1, -1), random.choice([9, 10, 11])),
        ('BACKGROUND', (0, 1), (-1, -1), body_bg),
        ('GRID', (0, 0), (-1, -1), random.choice([1, 2]), color_scheme['secondary']),
        ('TOPPADDING', (0, 0), (-1, -1), random.choice([10, 12, 14])),
        ('BOTTOMPADDING', (0, 0), (-1, -1), random.choice([10, 12, 14])),
    ]))
    
    story.append(contact_table)
    
    # Page break to next page
    story.append(PageBreak())
    
    # ===== PAGE 2: ABOUT US =====
    
    story.append(Spacer(1, random.choice([0.3, 0.5, 0.7])*inch))
    
    # Section title
    story.append(Paragraph("About Us", styles['SectionHeading']))
    
    # Vary if we show horizontal rule
    if random.random() > 0.2:
        story.append(HRFlowable(
            width="100%",
            thickness=random.choice([1, 2]),
            color=color_scheme['secondary'],
            spaceBefore=5,
            spaceAfter=15
        ))
    else:
        story.append(Spacer(1, 0.2*inch))
    
    # Company description
    story.append(Paragraph(description, styles['CustomBody']))
    story.append(Spacer(1, 0.3*inch))
    
    # Industry section
    story.append(Paragraph("Our Industry", styles['SectionHeading']))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=color_scheme['secondary'],
        spaceBefore=5,
        spaceAfter=15
    ))
    
    industry_text = f"We operate in the <b>{industry}</b> sector, where innovation and excellence drive our success. Our expertise allows us to deliver cutting-edge solutions that meet the evolving needs of our clients."
    story.append(Paragraph(industry_text, styles['CustomBody']))
    story.append(Spacer(1, 0.3*inch))
    
    # Our Services section
    story.append(Paragraph("Our Services", styles['SectionHeading']))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=color_scheme['secondary'],
        spaceBefore=5,
        spaceAfter=15
    ))
    
    services = [
        "Strategic Consulting & Advisory",
        "Custom Solution Development",
        "Technology Integration Services",
        "24/7 Customer Support & Maintenance",
        "Training & Knowledge Transfer"
    ]
    
    for service in services:
        bullet_text = f"• {service}"
        story.append(Paragraph(bullet_text, styles['CustomBody']))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Why Choose Us box
    why_choose_data = [
        [Paragraph('<b>Why Choose Us?</b>', styles['CustomBody'])],
        [Paragraph('We combine industry expertise with innovative solutions to deliver exceptional value to our clients.', styles['CustomBody'])],
        [Paragraph('Our team is dedicated to your success and committed to exceeding expectations.', styles['CustomBody'])],
    ]
    
    why_table = Table(why_choose_data, colWidths=[5.5*inch])
    why_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), color_scheme['secondary']),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, -1), color_scheme['accent']),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BOX', (0, 0), (-1, -1), 2, color_scheme['secondary']),
    ]))
    
    story.append(why_table)
    
    # Page break to next page
    story.append(PageBreak())
    
    # ===== PAGE 3: CONTACT & DETAILS =====
    
    story.append(Spacer(1, 0.5*inch))
    
    # Contact section
    story.append(Paragraph("Contact Us", styles['SectionHeading']))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=color_scheme['secondary'],
        spaceBefore=5,
        spaceAfter=15
    ))
    
    contact_intro = "We'd love to hear from you! Get in touch with us to learn more about how we can help your business succeed."
    story.append(Paragraph(contact_intro, styles['CustomBody']))
    story.append(Spacer(1, 0.2*inch))
    
    # Detailed contact table
    detailed_contact_data = [
        ['Contact Person', contact_person],
        ['Email Address', contact_email],
        ['Industry Focus', industry.title()],
        ['Company', company_name],
    ]
    
    detailed_table = Table(detailed_contact_data, colWidths=[2*inch, 3.5*inch])
    detailed_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), color_scheme['accent']),
        ('TEXTCOLOR', (0, 0), (0, -1), color_scheme['primary']),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, color_scheme['secondary']),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(detailed_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Call to action
    cta_style = ParagraphStyle(
        name='CTA',
        parent=styles['CustomBody'],
        fontSize=14,
        alignment=TA_CENTER,
        textColor=color_scheme['primary'],
        fontName='Helvetica-Bold',
        spaceBefore=20,
        spaceAfter=20
    )
    story.append(Paragraph("Ready to Get Started?", cta_style))
    story.append(Paragraph(f"Contact us today at <b>{contact_email}</b>", styles['CustomBody']))
    
    story.append(Spacer(1, 1*inch))
    
    # Footer box
    footer_data = [
        [Paragraph(f'<b>{company_name}</b><br/>Excellence in {industry.title()}', styles['CustomBody'])],
    ]
    
    footer_table = Table(footer_data, colWidths=[5.5*inch])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), color_scheme['primary']),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
    ]))
    
    story.append(footer_table)
    
    # Build PDF with custom canvas for page numbers
    doc.build(story, canvasmaker=lambda *args, **kwargs: NumberedCanvas(*args, company_name=company_name, **kwargs))


def generate_all_pdf_brochures(df, output_dir='output/pdf_brochures'):
    """
    Generates PDF brochures for all companies in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with client_data column
        output_dir (str): Directory to save all PDFs
        
    Returns:
        pd.DataFrame: Original DataFrame with added PDF path column
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_paths = []
    
    print(f"Generating PDF brochures for {len(df)} companies...")
    
    for idx, row in df.iterrows():
        client_data = row['client_data']
        pdf_path = os.path.join(output_dir, f'brochure_{idx:03d}.pdf')
        
        generate_pdf_brochure(client_data, pdf_path)
        pdf_paths.append(pdf_path)
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Generated PDF brochures for {idx + 1}/{len(df)} companies")
    
    df['pdf_brochure_path'] = pdf_paths
    
    print(f"✓ All PDF brochures saved to: {os.path.abspath(output_dir)}")
    
    return df
