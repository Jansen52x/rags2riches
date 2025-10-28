"""
Image Generator Module
Generates synthetic marketing materials (brochures, flyers) using Pillow.
"""
from PIL import Image, ImageDraw, ImageFont
import os
import random
import textwrap


# Color palettes for different designs
COLOR_PALETTES = [
    {'bg': '#1a237e', 'accent': '#00bcd4', 'text': '#ffffff', 'secondary': '#e3f2fd'},  # Blue
    {'bg': '#004d40', 'accent': '#00bfa5', 'text': '#ffffff', 'secondary': '#e0f2f1'},  # Teal
    {'bg': '#311b92', 'accent': '#7c4dff', 'text': '#ffffff', 'secondary': '#ede7f6'},  # Purple
    {'bg': '#b71c1c', 'accent': '#ff5252', 'text': '#ffffff', 'secondary': '#ffebee'},  # Red
    {'bg': '#e65100', 'accent': '#ff9800', 'text': '#ffffff', 'secondary': '#fff3e0'},  # Orange
    {'bg': '#1b5e20', 'accent': '#4caf50', 'text': '#ffffff', 'secondary': '#e8f5e9'},  # Green
    {'bg': '#f5f5f5', 'accent': '#607d8b', 'text': '#212121', 'secondary': '#ffffff'},  # Light gray
    {'bg': '#263238', 'accent': '#ffab40', 'text': '#eceff1', 'secondary': '#37474f'},  # Dark blue-gray
    {'bg': '#6a1b9a', 'accent': '#ce93d8', 'text': '#f3e5f5', 'secondary': '#4a148c'},  # Deep purple
    {'bg': '#d32f2f', 'accent': '#ffcdd2', 'text': '#ffebee', 'secondary': '#b71c1c'},  # Deep red
]

# "Challenging" palettes with poor contrast (for RAG training)
CHALLENGING_PALETTES = [
    {'bg': '#2c2c2c', 'accent': '#1a1a1a', 'text': '#333333', 'secondary': '#3a3a3a'},  # Dark on dark
    {'bg': '#f0f0f0', 'accent': '#e8e8e8', 'text': '#ffffff', 'secondary': '#fafafa'},  # Light on light
    {'bg': '#ff5252', 'accent': '#ff1744', 'text': '#ff6666', 'secondary': '#ff8a80'},  # Red on red
    {'bg': '#2196f3', 'accent': '#1976d2', 'text': '#42a5f5', 'secondary': '#64b5f6'},  # Blue on blue
    {'bg': '#4caf50', 'accent': '#388e3c', 'text': '#66bb6a', 'secondary': '#81c784'},  # Green on green
]

# Layout variations
LAYOUT_STYLES = ['centered', 'left_aligned', 'minimal', 'decorated', 'modern']


def get_default_font(size):
    """
    Gets a font with fallback options for cross-platform compatibility.
    
    Args:
        size (int): Font size
        
    Returns:
        ImageFont: Font object
    """
    font_options = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "Helvetica.ttf",
    ]
    
    for font_name in font_options:
        try:
            return ImageFont.truetype(font_name, size)
        except:
            continue
    
    # Fallback to default font
    return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    """
    Wraps text to fit within a specified width.
    
    Args:
        text (str): Text to wrap
        font: Font object
        max_width (int): Maximum width in pixels
        draw: ImageDraw object
        
    Returns:
        list: List of wrapped text lines
    """
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def generate_company_brochure(client_data, output_path, width=800, height=1000):
    """
    Generates a company brochure image with varied layouts and styling.
    
    Args:
        client_data (dict): Dictionary containing company information
        output_path (str): Path to save the image
        width (int): Image width in pixels
        height (int): Image height in pixels
    """
    # Create image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 20% chance to use challenging palette, 80% normal
    if random.random() < 0.2:
        palette = random.choice(CHALLENGING_PALETTES)
    else:
        palette = random.choice(COLOR_PALETTES)
    
    # Select random layout style
    layout_style = random.choice(LAYOUT_STYLES)
    
    # Vary header height
    header_height = random.choice([height // 4, height // 3, height // 5])
    draw.rectangle([(0, 0), (width, header_height)], fill=palette['bg'])
    
    # Add company name
    font_title = get_default_font(random.choice([40, 48, 56]))  # Vary font size
    font_subtitle = get_default_font(random.choice([18, 20, 24]))
    font_body = get_default_font(random.choice([16, 18, 20]))
    font_small = get_default_font(random.choice([12, 14, 16]))
    
    company_name = client_data.get('company_name', 'Company Name')
    industry = client_data.get('industry', 'Industry')
    
    # Apply layout style
    if layout_style == 'centered':
        # Center company name in header
        bbox = draw.textbbox((0, 0), company_name, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, 50), company_name, fill=palette['text'], font=font_title)
        
        # Add industry subtitle
        bbox = draw.textbbox((0, 0), industry.upper(), font=font_subtitle)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, 120), industry.upper(), fill=palette['accent'], font=font_subtitle)
    elif layout_style == 'left_aligned':
        # Left-aligned text
        margin = 60
        draw.text((margin, 50), company_name, fill=palette['text'], font=font_title)
        draw.text((margin, 120), industry.upper(), fill=palette['accent'], font=font_subtitle)
    elif layout_style == 'minimal':
        # Minimal centered style with more spacing
        bbox = draw.textbbox((0, 0), company_name, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, header_height // 2 - 30), company_name, fill=palette['text'], font=font_title)
    elif layout_style == 'decorated':
        # Centered with decorative elements
        bbox = draw.textbbox((0, 0), company_name, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        # Add background rectangle behind text
        padding = 20
        draw.rectangle([(x - padding, 40), (x + text_width + padding, 100)], fill=palette['accent'])
        draw.text((x, 50), company_name, fill=palette['bg'], font=font_title)
        
        bbox = draw.textbbox((0, 0), industry.upper(), font=font_subtitle)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, 120), industry.upper(), fill=palette['text'], font=font_subtitle)
    else:  # 'modern'
        # Modern style with offset
        margin = 80
        draw.text((margin, 60), company_name, fill=palette['text'], font=font_title)
        draw.text((margin + 10, 130), industry.upper(), fill=palette['accent'], font=font_subtitle)
    
    # Randomly add or skip decorative line
    if random.random() > 0.3:
        line_width = random.choice([width//4, width//3, 3*width//4])
        line_x = (width - line_width) // 2
        draw.rectangle([(line_x, header_height - 20), (line_x + line_width, header_height - 15)], 
                       fill=palette['accent'])
    
    # Content section
    y_pos = header_height + 60
    margin = random.choice([40, 60, 80])  # Vary margins
    
    # About Us section - vary text color (sometimes match bg for challenge)
    section_text_color = palette['bg'] if random.random() > 0.1 else palette['accent']
    draw.text((margin, y_pos), "ABOUT US", fill=section_text_color, font=font_subtitle)
    y_pos += 40
    
    description = client_data.get('company_description', 'Company description goes here.')
    wrapped_desc = wrap_text(description, font_body, width - 2*margin, draw)
    
    # Vary body text color - 10% chance to use very low contrast
    body_text_color = '#333333'
    if random.random() < 0.1:
        body_text_color = palette['secondary']  # Low contrast
    
    for line in wrapped_desc[:random.choice([3, 4, 5])]:  # Vary number of lines
        draw.text((margin, y_pos), line, fill=body_text_color, font=font_body)
        y_pos += random.choice([28, 30, 35])  # Vary line spacing
    
    y_pos += random.choice([30, 40, 50])
    
    # Contact Information section - vary box style
    box_height = random.choice([180, 200, 220])
    box_bg = palette['secondary'] if random.random() > 0.15 else palette['bg']  # 15% chance dark box
    box_text_color = '#333333' if box_bg == palette['secondary'] else palette['text']
    
    draw.rectangle([(margin, y_pos), (width - margin, y_pos + box_height)], 
                   fill=box_bg, outline=palette['accent'], width=random.choice([1, 2, 3]))
    y_pos += 30
    
    draw.text((margin + 20, y_pos), "CONTACT INFORMATION", fill=palette['accent'], font=font_subtitle)
    y_pos += 50
    
    contact_person = client_data.get('contact_person', 'Contact Person')
    contact_email = client_data.get('contact_email', 'email@example.com')
    
    draw.text((margin + 20, y_pos), f"Contact Person: {contact_person}", 
              fill=box_text_color, font=font_body)
    y_pos += 35
    draw.text((margin + 20, y_pos), f"Email: {contact_email}", 
              fill=box_text_color, font=font_body)
    y_pos += 35
    draw.text((margin + 20, y_pos), f"Industry: {industry}", 
              fill=box_text_color, font=font_body)
    
    # Footer
    footer_y = height - 60
    draw.rectangle([(0, footer_y), (width, height)], fill=palette['bg'])
    footer_text = f"© 2025 {company_name}. All rights reserved."
    bbox = draw.textbbox((0, 0), footer_text, font=font_small)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, footer_y + 20), footer_text, fill=palette['text'], font=font_small)
    
    # Save image
    img.save(output_path, 'PNG')


def generate_company_flyer(client_data, output_path, width=600, height=800):
    """
    Generates a marketing flyer image with varied layouts and styling (more compact than brochure).
    
    Args:
        client_data (dict): Dictionary containing company information
        output_path (str): Path to save the image
        width (int): Image width in pixels
        height (int): Image height in pixels
    """
    # Create image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 20% chance to use challenging palette
    if random.random() < 0.2:
        palette = random.choice(CHALLENGING_PALETTES)
    else:
        palette = random.choice(COLOR_PALETTES)
    
    # Select random layout
    layout_style = random.choice(LAYOUT_STYLES)
    
    # Draw background with accent color - vary background
    bg_color = palette['secondary'] if random.random() > 0.2 else palette['accent']
    draw.rectangle([(0, 0), (width, height)], fill=bg_color)
    
    # Draw top banner - vary height
    banner_height = random.choice([120, 150, 180])
    draw.rectangle([(0, 0), (width, banner_height)], fill=palette['bg'])
    
    # 70% chance to add diagonal accent stripe
    if random.random() > 0.3:
        stripe_offset = random.choice([30, 40, 50])
        points = [(0, banner_height), (width, banner_height), 
                  (width, banner_height + stripe_offset), (0, banner_height + stripe_offset - 20)]
        draw.polygon(points, fill=palette['accent'])
    
    # Add company name with varied fonts
    font_title = get_default_font(random.choice([36, 40, 44]))
    font_subtitle = get_default_font(random.choice([16, 18, 20]))
    font_body = get_default_font(random.choice([14, 16, 18]))
    font_cta = get_default_font(random.choice([22, 24, 26]))
    
    company_name = client_data.get('company_name', 'Company Name')
    industry = client_data.get('industry', 'Industry')
    
    # Apply layout style to title
    if layout_style in ['centered', 'minimal']:
        # Center company name
        bbox = draw.textbbox((0, 0), company_name, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y_title = random.choice([30, 40, 50])
        draw.text((x, y_title), company_name, fill=palette['text'], font=font_title)
        
        # Add industry
        bbox = draw.textbbox((0, 0), industry, font=font_subtitle)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        # 15% chance text color matches bg
        industry_color = palette['accent'] if random.random() > 0.15 else palette['bg']
        draw.text((x, y_title + 55), industry, fill=industry_color, font=font_subtitle)
    else:
        # Left or right aligned
        margin = random.choice([30, 40, 50])
        y_title = random.choice([30, 40, 50])
        draw.text((margin, y_title), company_name, fill=palette['text'], font=font_title)
        draw.text((margin, y_title + 55), industry, fill=palette['accent'], font=font_subtitle)
    
    # Content area
    y_pos = banner_height + random.choice([60, 80, 100])
    margin = random.choice([30, 40, 50])
    
    # Company description - vary text color (10% low contrast)
    description = client_data.get('company_description', 'Company description.')
    wrapped_desc = wrap_text(description, font_body, width - 2*margin, draw)
    
    desc_color = '#333333' if random.random() > 0.1 else bg_color
    
    for line in wrapped_desc[:random.choice([3, 4, 5])]:
        draw.text((margin, y_pos), line, fill=desc_color, font=font_body)
        y_pos += random.choice([24, 28, 32])
    
    y_pos += random.choice([40, 50, 60])
    
    # Call-to-action box - vary style
    cta_height = random.choice([160, 180, 200])
    cta_y = height - cta_height - 40
    cta_bg = 'white' if random.random() > 0.2 else palette['bg']
    cta_border_color = palette['accent'] if cta_bg == 'white' else palette['secondary']
    cta_border_width = random.choice([2, 3, 4])
    
    draw.rectangle([(margin, cta_y), (width - margin, cta_y + cta_height)], 
                   fill=cta_bg, outline=cta_border_color, width=cta_border_width)
    
    # CTA text - vary color based on background
    cta_text = random.choice(["Get in Touch!", "Contact Us!", "Let's Talk!", "Reach Out!"])
    cta_text_color = palette['bg'] if cta_bg == 'white' else palette['text']
    bbox = draw.textbbox((0, 0), cta_text, font=font_cta)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, cta_y + 20), cta_text, fill=cta_text_color, font=font_cta)
    
    # Contact info
    contact_email = client_data.get('contact_email', 'email@example.com')
    contact_person = client_data.get('contact_person', 'Contact Person')
    
    bbox = draw.textbbox((0, 0), contact_email, font=font_body)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, cta_y + 80), contact_email, fill=palette['accent'], font=font_body)
    
    bbox = draw.textbbox((0, 0), contact_person, font=font_body)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, cta_y + 110), contact_person, fill='#666666', font=font_body)
    
    # Save image
    img.save(output_path, 'PNG')


def generate_marketing_materials(client_data, output_dir, company_id):
    """
    Generates both brochure and flyer for a company.
    
    Args:
        client_data (dict): Dictionary containing company information
        output_dir (str): Directory to save images
        company_id (int): Unique identifier for the company
        
    Returns:
        dict: Paths to generated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file paths
    brochure_path = os.path.join(output_dir, f'brochure_{company_id:03d}.png')
    flyer_path = os.path.join(output_dir, f'flyer_{company_id:03d}.png')
    
    # Generate images
    generate_company_brochure(client_data, brochure_path)
    generate_company_flyer(client_data, flyer_path)
    
    return {
        'brochure': brochure_path,
        'flyer': flyer_path
    }


def generate_all_materials(df, output_dir='output/marketing_materials'):
    """
    Generates marketing materials for all companies in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with client_data column
        output_dir (str): Directory to save all images
        
    Returns:
        pd.DataFrame: Original DataFrame with added image path columns
    """
    brochure_paths = []
    flyer_paths = []
    
    print(f"Generating marketing materials for {len(df)} companies...")
    
    for idx, row in df.iterrows():
        client_data = row['client_data']
        paths = generate_marketing_materials(client_data, output_dir, idx)
        brochure_paths.append(paths['brochure'])
        flyer_paths.append(paths['flyer'])
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Generated materials for {idx + 1}/{len(df)} companies")
    
    df['brochure_path'] = brochure_paths
    df['flyer_path'] = flyer_paths
    
    print(f"✓ All marketing materials saved to: {os.path.abspath(output_dir)}")
    
    return df
