# Image & PDF Variations for RAG Training

## Overview
This generator creates diverse marketing materials with intentional variations to improve RAG system robustness.

## Image Variations (Brochures & Flyers)

### Color Palettes
- **10 Standard Palettes:** Professional color combinations (Blue, Teal, Purple, Red, Orange, Green, Gray, etc.)
- **5 Challenging Palettes (20% chance):** Poor contrast for robust training
  - Dark on dark (text barely visible)
  - Light on light (low contrast)
  - Red on red variations
  - Blue on blue variations
  - Green on green variations

### Layout Styles (Random per image)
1. **Centered:** Traditional centered text alignment
2. **Left Aligned:** Modern left-aligned layout
3. **Minimal:** Clean, lots of white space
4. **Decorated:** Enhanced with background boxes and shapes
5. **Modern:** Asymmetric with offset elements

### Variable Elements

#### Brochures (800x1000px)
- Header height: 25%, 33%, or 20% of page
- Font sizes: Title (40-56px), Subtitle (18-24px), Body (16-20px)
- Margins: 40px, 60px, or 80px
- Line spacing: 28px, 30px, or 35px
- Text lines displayed: 3, 4, or 5 lines
- Section spacing: 30px, 40px, or 50px
- Contact box height: 180px, 200px, or 220px
- Border width: 1px, 2px, or 3px
- **10% chance:** Body text matches background (very low contrast)
- **15% chance:** Contact box uses dark background
- **30% chance:** No decorative line

#### Flyers (600x800px)
- Banner height: 120px, 150px, or 180px
- Background: Secondary color (80%) or accent color (20%)
- Font sizes: Title (36-44px), Subtitle (16-20px), Body (14-18px), CTA (22-26px)
- Margins: 30px, 40px, or 50px
- Stripe offset: 30px, 40px, or 50px
- Description lines: 3, 4, or 5 lines
- Line spacing: 24px, 28px, or 32px
- CTA height: 160px, 180px, or 200px
- CTA background: White (80%) or primary color (20%)
- CTA text: "Get in Touch!", "Contact Us!", "Let's Talk!", or "Reach Out!"
- **30% chance:** No diagonal stripe
- **10% chance:** Description text has very low contrast
- **15% chance:** Industry text matches background color
- **20% chance:** Dark CTA box

## PDF Variations (Multi-page Brochures)

### Color Schemes
- **7 Standard Schemes:** Professional color combinations
- **3 Challenging Schemes (20% chance):** Poor contrast
  - Dark gray monochrome
  - Light gray monochrome
  - Blue monochrome

### Layout Styles (Random per PDF)
1. **Standard:** Traditional professional layout
2. **Modern:** Contemporary styling
3. **Minimal:** Clean and simple
4. **Bold:** Larger fonts and prominent elements

### Variable Elements Per Layout

#### Font Sizes by Layout
- **Bold:** 32px title, 18px subtitle, 16px heading, 12px body
- **Minimal:** 24px title, 14px subtitle, 12px heading, 10px body
- **Standard/Modern:** 28px title, 16px subtitle, 14px heading, 11px body

#### Page 1 (Cover)
- Top margin: 1.2in, 1.5in, or 1.8in
- Spacing after title: 0.2in, 0.3in, or 0.4in
- Spacing after subtitle: 0.3in, 0.5in, or 0.7in
- HR line width: 60%, 80%, or 100%
- HR thickness: 1px, 2px, or 3px
- Tagline font: 11px, 13px, or 15px
- Bottom spacing: 1.5in, 2in, or 2.5in
- Contact table font: 11px, 12px, or 13px
- Cell padding: 10px, 12px, or 14px
- **20% chance:** No horizontal rule
- **15% chance:** Tagline uses accent color (low contrast)
- **20% chance:** Contact box header uses secondary color
- **15% chance:** Contact box body uses white background

#### Page 2 (About)
- Top margin: 0.3in, 0.5in, or 0.7in
- HR thickness: 1px or 2px
- **20% chance:** No horizontal rule after section titles

#### Table Styling
- Grid line width: 1px or 2px
- Cell padding: 10px, 12px, or 14px
- Header background: Primary or secondary color
- Body background: Accent color or white

## Challenging Cases for RAG Training

### Low Contrast Scenarios (~20% of all outputs)
1. **Same color text and background** (very difficult to read)
2. **Monochrome palettes** (no color differentiation)
3. **Similar tones** (minimal contrast ratios)

### Layout Variations
- Different alignment styles
- Variable spacing and margins
- Optional decorative elements
- Multiple font size combinations

## Benefits for RAG Systems

1. **Robustness:** Handles poorly formatted documents
2. **Flexibility:** Adapts to various visual styles
3. **Real-world simulation:** Mimics inconsistent real documents
4. **OCR challenges:** Tests text extraction on difficult cases
5. **Generalization:** Prevents overfitting to single format

## Statistics

Per 75 companies generated:
- **Standard designs:** ~60 (80%)
- **Challenging designs:** ~15 (20%)
- **Unique layouts:** 5 different styles randomly assigned
- **Color variations:** 15 total palettes (10 standard + 5 challenging)
- **Total variations possible:** 1000+ unique combinations

## Running the Generator

```bash
python synthetic-data\generate_data.py
```

Configure in `generate_data.py`:
- `NUM_RECORDS`: Number of companies to generate
- `GENERATE_IMAGES`: True/False for PNG brochures/flyers
- `GENERATE_PDFS`: True/False for PDF brochures
- `CLEAN_OUTPUT`: True/False to overwrite existing files
