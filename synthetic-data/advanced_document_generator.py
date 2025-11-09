"""
Advanced Document Generator Module
Generates diverse document types for each company including:
- Multiple brochures (marketing, product, services)
- Financial reports (quarterly, annual)
- Press releases
- Advertisements
- Partnership documents (shared between companies)
- Internal memos
- Case studies

Supports two modes:
- Gemini AI: Generates realistic, coherent content for real companies (from seed CSV)
- Faker: Generates synthetic content for fake companies
"""
from faker import Faker
import random
import time

fake = Faker()

# Import Gemini model from data_generator
try:
    from data_generator import gemini_model, mark_gemini_failed
except ImportError:
    gemini_model = None
    def mark_gemini_failed():
        pass


def _generate_with_gemini(prompt, fallback_content, is_seed_company=False):
    """
    Helper to generate content with Gemini AI with Google Search Grounding.
    For real companies: Auto-retry with exponential backoff on rate limits.
    For synthetic companies: Fallback to Faker on errors.
    
    Args:
        prompt (str): The prompt to send to Gemini
        fallback_content (str): Content to use if Gemini fails (only for synthetic companies)
        is_seed_company (bool): If True, auto-retry on rate limits (never use fallback)
        
    Returns:
        str: Generated content from Gemini or fallback
        
    Raises:
        Exception: If Gemini fails after retries for seed companies
    """
    if gemini_model is None:
        if is_seed_company:
            raise Exception("Gemini not available but required for seed company")
        return fallback_content
    
    max_retries = 5 if is_seed_company else 0  # Retry for real companies only
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Generate content with Gemini (grounding via prompt instructions)
            response = gemini_model.generate_content(prompt)
            time.sleep(4.0)  # Rate limit delay (60s / 15 = 4s per request)
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_msg or "Resource exhausted" in error_msg:
                if is_seed_company and retry_count < max_retries:
                    # Auto-retry for real companies with exponential backoff
                    wait_time = 60 * (2 ** retry_count)  # 60s, 120s, 240s, 480s, 960s
                    retry_count += 1
                    print(f"      ⏳ Rate limit hit. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                    time.sleep(wait_time)
                    continue  # Retry the request
                elif is_seed_company:
                    # Exhausted all retries for real company
                    print(f"      ❌ Failed after {max_retries} retries: {e}")
                    raise
                else:
                    # Synthetic company - use fallback
                    print(f"      ⚠️  Gemini error (using Faker fallback): {e}")
                    return fallback_content
            else:
                # Non-rate-limit error
                if is_seed_company:
                    print(f"      ❌ Gemini API error for real company: {e}")
                    raise
                else:
                    print(f"      ⚠️  Gemini error (using Faker fallback): {e}")
                    return fallback_content
    
    # Should never reach here, but just in case
    if is_seed_company:
        raise Exception(f"Failed to generate content after {max_retries} retries")
    return fallback_content


def generate_product_brochure(client_data, use_gemini=False, variation='A'):
    """Generates a product-focused brochure.
    
    Args:
        client_data (dict): Company information
        use_gemini (bool): If True, use Gemini AI for content generation
        variation (str): 'A' or 'B' to generate different product focus for variety
    """
    company_name = client_data.get('company_name', 'Company')
    industry = client_data.get('industry', 'Industry')
    company_desc = client_data.get('company_description', '')
    
    # Faker-based product names (used for both modes)
    products = [
        f"{fake.word().title()} {fake.word().title()} System",
        f"{fake.color_name().title()} {fake.word().title()} Platform",
        f"Premium {fake.word().title()} Suite",
        f"Enterprise {fake.word().title()} Solution"
    ]
    
    product_name = random.choice(products)
    
    # Generate content with Gemini or Faker
    if use_gemini and gemini_model is not None:
        # Different prompts for variation A (image) and B (PDF)
        variation_instruction = ""
        if variation == 'A':
            variation_instruction = "Focus on ONE of their flagship or primary products/services."
        else:  # variation == 'B'
            variation_instruction = "Focus on a DIFFERENT product/service line. If they have multiple offerings, choose a secondary, complementary, or enterprise-tier product that's distinct from their main consumer offering."
        
        prompt = f"""You are writing a professional product brochure for {company_name}.

Company: {company_name}
Industry: {industry}
Description: {company_desc}

IMPORTANT: {variation_instruction}

Write a detailed, realistic product brochure (250-300 words) that includes:
1. A REAL product/service that {company_name} actually offers (use your knowledge of this company)
2. Specific features and capabilities (be concrete and detailed)
3. Target customers and use cases
4. Competitive advantages and unique value proposition
5. Technical specifications or key metrics where applicable

Be specific - mention actual product names, real features, and genuine capabilities of {company_name}.
Make it sound like an actual {company_name} marketing document.
Write in a professional, compelling tone."""
        
        fallback_content = (
            f"Introducing {product_name} by {company_name}. "
            f"{fake.catch_phrase()}. "
            f"Our innovative solution in the {industry} sector delivers {fake.bs().lower()}. "
            f"Key features include: {fake.catch_phrase().lower()}, {fake.bs().lower()}, "
            f"and {fake.catch_phrase().lower()}. "
            f"{fake.text(max_nb_chars=150)} "
            f"Contact us today to learn more about how {product_name} can transform your business."
        )
        
        is_seed = client_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = (
            f"Introducing {product_name} by {company_name}. "
            f"{fake.catch_phrase()}. "
            f"Our innovative solution in the {industry} sector delivers {fake.bs().lower()}. "
            f"Key features include: {fake.catch_phrase().lower()}, {fake.bs().lower()}, "
            f"and {fake.catch_phrase().lower()}. "
            f"{fake.text(max_nb_chars=150)} "
            f"Contact us today to learn more about how {product_name} can transform your business."
        )
    
    return {
        'document_type': 'product_brochure',
        'title': f'{product_name} - Product Brochure',
        'company_name': company_name,
        'content': content,
        'product_name': product_name
    }


def generate_services_brochure(client_data, use_gemini=False, variation='A'):
    """Generates a services-focused brochure.
    
    Args:
        client_data (dict): Company information
        use_gemini (bool): If True, use Gemini AI for content generation
        variation (str): 'A' or 'B' to generate different service focus for variety
    """
    company_name = client_data.get('company_name', 'Company')
    industry = client_data.get('industry', 'Industry')
    company_desc = client_data.get('company_description', '')
    
    services = [
        f"{fake.word().title()} Consulting",
        f"{fake.word().title()} Management Services",
        f"Professional {fake.word().title()} Support",
        f"{fake.word().title()} Implementation"
    ]
    
    if use_gemini and gemini_model is not None:
        # Different prompts for variation A (image) and B (PDF)
        variation_instruction = ""
        if variation == 'A':
            variation_instruction = "Focus on their core service offerings or primary business services."
        else:  # variation == 'B'
            variation_instruction = "Focus on DIFFERENT or secondary service lines. If they offer multiple service categories (e.g., consulting, support, implementation), highlight services that complement but differ from their main offerings."
        
        prompt = f"""You are writing a professional services brochure for {company_name}.

Company: {company_name}
Industry: {industry}
Description: {company_desc}

IMPORTANT: {variation_instruction}

Write a detailed services brochure (250-300 words) that includes:
1. REAL services that {company_name} actually provides (use your knowledge of this company)
2. Specific service offerings with concrete details
3. Target market and client types
4. Service delivery approach and methodologies
5. Success metrics, certifications, or partnerships (if applicable for this company)

Be specific - mention actual services, real capabilities, and genuine expertise areas of {company_name}.
Make it sound like an actual {company_name} services document.
Write in a professional, authoritative tone."""
        
        fallback_content = (
            f"{company_name} offers comprehensive services in {industry}. "
            f"Our service portfolio includes: {', '.join(random.sample(services, 3))}. "
            f"{fake.catch_phrase()}. "
            f"With over {random.randint(5, 20)} years of experience, we provide {fake.bs().lower()}. "
            f"{fake.text(max_nb_chars=120)} "
            f"Our team of experts ensures {fake.catch_phrase().lower()} for all our clients."
        )
        
        is_seed = client_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = (
            f"{company_name} offers comprehensive services in {industry}. "
            f"Our service portfolio includes: {', '.join(random.sample(services, 3))}. "
            f"{fake.catch_phrase()}. "
            f"With over {random.randint(5, 20)} years of experience, we provide {fake.bs().lower()}. "
            f"{fake.text(max_nb_chars=120)} "
            f"Our team of experts ensures {fake.catch_phrase().lower()} for all our clients."
        )
    
    return {
        'document_type': 'services_brochure',
        'title': f'{company_name} - Services Overview',
        'company_name': company_name,
        'content': content
    }


def generate_financial_report(client_data, quarter=None, year=None, use_gemini=False):
    """Generates a quarterly or annual financial report.
    
    Args:
        client_data (dict): Company information
        quarter (str): Quarter identifier (Q1, Q2, Q3, Q4, or Annual)
        year (int): Report year
        use_gemini (bool): If True, use Gemini AI for content generation
    """
    company_name = client_data.get('company_name', 'Company')
    industry = client_data.get('industry', 'Industry')
    
    if year is None:
        year = random.randint(2022, 2025)
    
    if quarter is None:
        quarter = random.choice(['Q1', 'Q2', 'Q3', 'Q4', 'Annual'])
    
    revenue = random.randint(500000, 50000000)
    growth = random.uniform(-5, 25)
    profit_margin = random.uniform(5, 30)
    
    if use_gemini and gemini_model is not None:
        prompt = f"""You are writing a realistic financial report for {company_name}.

Company: {company_name}
Industry: {industry}
Period: {quarter} {year}

Write a professional financial report (300-350 words) that includes:
1. Executive summary with realistic financial performance for {company_name}
2. Revenue figures and growth metrics (be specific - use your knowledge of {company_name}'s actual scale)
3. Key business highlights and achievements from this period
4. Market conditions and competitive landscape in the {industry} sector
5. Forward-looking statements and guidance

Base the numbers and narrative on what would be realistic for {company_name} given its actual size and business model.
Include specific business segments, product lines, or geographic regions where applicable.
Make it sound like an actual {company_name} investor relations document.
Write in a formal, professional tone suitable for investors and analysts."""
        
        fallback_content = (
            f"{company_name} Financial Report - {quarter} {year}. "
            f"Revenue: ${revenue:,.2f}. Year-over-year growth: {growth:.1f}%. "
            f"Operating profit margin: {profit_margin:.1f}%. "
            f"Executive Summary: {fake.text(max_nb_chars=100)} "
            f"Key Highlights: {fake.catch_phrase()}. {fake.bs().title()}. "
            f"Market conditions: {fake.text(max_nb_chars=80)} "
            f"Future outlook: {fake.catch_phrase()}. {fake.bs().title()}."
        )
        
        is_seed = client_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = (
            f"{company_name} Financial Report - {quarter} {year}. "
            f"Revenue: ${revenue:,.2f}. Year-over-year growth: {growth:.1f}%. "
            f"Operating profit margin: {profit_margin:.1f}%. "
            f"Executive Summary: {fake.text(max_nb_chars=100)} "
            f"Key Highlights: {fake.catch_phrase()}. {fake.bs().title()}. "
            f"Market conditions: {fake.text(max_nb_chars=80)} "
            f"Future outlook: {fake.catch_phrase()}. {fake.bs().title()}."
        )
    
    return {
        'document_type': 'financial_report',
        'title': f'{company_name} - {quarter} {year} Financial Report',
        'company_name': company_name,
        'quarter': quarter,
        'year': year,
        'content': content,
        'revenue': revenue,
        'growth_rate': growth,
        'profit_margin': profit_margin
    }


def generate_press_release(client_data, use_gemini=False):
    """Generates a press release.
    
    Args:
        client_data (dict): Company information
        use_gemini (bool): If True, use Gemini AI for content generation
    """
    company_name = client_data.get('company_name', 'Company')
    contact_person = client_data.get('contact_person', 'Contact')
    contact_email = client_data.get('contact_email', 'email@example.com')
    industry = client_data.get('industry', 'Industry')
    company_desc = client_data.get('company_description', '')
    
    topics = [
        f"Announces Strategic Partnership",
        f"Launches Revolutionary New Product",
        f"Expands Operations",
        f"Achieves Major Milestone",
        f"Wins Industry Award",
        f"Appoints New Leadership"
    ]
    
    topic = random.choice(topics)
    
    if use_gemini and gemini_model is not None:
        prompt = f"""You are writing a professional press release for {company_name}.

Company: {company_name}
Industry: {industry}
Description: {company_desc}
Contact: {contact_person} ({contact_email})

Write a realistic press release (250-300 words) announcing a significant company development. 

Choose ONE realistic announcement that {company_name} might actually make:
- Product launch (use actual {company_name} product type)
- Partnership announcement (with realistic partner)
- Financial milestone or earnings report
- Expansion or new market entry
- Leadership appointment or organizational change
- Award or recognition

Include:
1. "FOR IMMEDIATE RELEASE" header
2. Compelling headline with location and date
3. Opening paragraph with the key announcement
4. Supporting details and context (2-3 paragraphs)
5. A quote from a realistic executive (CEO, VP, etc.)
6. Company boilerplate paragraph
7. Contact information

Base the announcement on something that would be realistic and newsworthy for {company_name}.
Use industry-appropriate language and tone.
Make it sound like an actual {company_name} press release."""
        
        fallback_content = (
            f"FOR IMMEDIATE RELEASE. "
            f"{company_name} {topic}. "
            f"{fake.city()}, {fake.date_this_year().strftime('%B %d, %Y')} - "
            f"{company_name} today announced {fake.catch_phrase().lower()}. "
            f"{fake.text(max_nb_chars=150)} "
            f'"{fake.catch_phrase()}," said {contact_person}, spokesperson for {company_name}. '
            f"{fake.text(max_nb_chars=100)} "
            f"For more information, contact {contact_email}."
        )
        
        is_seed = client_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = (
            f"FOR IMMEDIATE RELEASE. "
            f"{company_name} {topic}. "
            f"{fake.city()}, {fake.date_this_year().strftime('%B %d, %Y')} - "
            f"{company_name} today announced {fake.catch_phrase().lower()}. "
            f"{fake.text(max_nb_chars=150)} "
            f'"{fake.catch_phrase()}," said {contact_person}, spokesperson for {company_name}. '
            f"{fake.text(max_nb_chars=100)} "
            f"For more information, contact {contact_email}."
        )
    
    return {
        'document_type': 'press_release',
        'title': f'{company_name} {topic}',
        'company_name': company_name,
        'content': content,
        'contact_person': contact_person,
        'contact_email': contact_email
    }


def generate_advertisement(client_data, use_gemini=False):
    """Generates an advertisement.
    
    Args:
        client_data (dict): Company information
        use_gemini (bool): If True, use Gemini AI for content generation
    """
    company_name = client_data.get('company_name', 'Company')
    industry = client_data.get('industry', 'Industry')
    
    headlines = [
        f"Transform Your Business Today",
        f"The Future of {industry.split()[0].title()} is Here",
        f"Experience Excellence",
        f"Innovation You Can Trust",
        f"Leading the Way in {industry.split()[0].title()}"
    ]
    
    headline = random.choice(headlines)
    
    # Faker fallback content
    fallback_content = (
        f"{headline}! "
        f"{company_name} - {fake.catch_phrase()}. "
        f"Discover how we're {fake.bs().lower()}. "
        f"Why choose us? {fake.catch_phrase()}. {fake.catch_phrase()}. {fake.catch_phrase()}. "
        f"{fake.text(max_nb_chars=80)} "
        f"Don't wait! Contact us today and join the revolution in {industry}."
    )
    
    if use_gemini and gemini_model is not None:
        prompt = f"""You are writing a professional advertisement for {company_name}.

Company: {company_name}
Industry: {industry}

Write a compelling advertisement (200-250 words) that includes:
1. An attention-grabbing headline
2. Clear value proposition highlighting {company_name}'s actual strengths
3. Specific benefits or features (use real {company_name} offerings)
4. Target audience identification
5. Social proof or credibility indicators (awards, customer count, years in business, etc. - use realistic info for {company_name})
6. Strong call-to-action

Make it persuasive and customer-focused.
Use language and messaging that {company_name} would actually use in their marketing.
Include specific, concrete benefits rather than generic claims.
Write in an engaging, professional tone."""
        
        is_seed = client_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = fallback_content
    
    return {
        'document_type': 'advertisement',
        'title': f'{company_name} - Advertisement',
        'company_name': company_name,
        'content': content
    }


def generate_case_study(client_data, use_gemini=False):
    """Generates a case study.
    
    Args:
        client_data (dict): Company information
        use_gemini (bool): If True, use Gemini AI for content generation
    """
    company_name = client_data.get('company_name', 'Company')
    
    client_names = [fake.company() for _ in range(3)]
    client_name = random.choice(client_names)
    
    improvement = random.randint(15, 85)
    metric = random.choice(['efficiency', 'revenue', 'productivity', 'customer satisfaction', 'market share'])
    
    fallback_content = (
        f"Case Study: How {client_name} Achieved Success with {company_name}. "
        f"Challenge: {fake.text(max_nb_chars=80)} "
        f"Solution: {company_name} implemented {fake.catch_phrase().lower()} to help {client_name} {fake.bs().lower()}. "
        f"Results: {improvement}% improvement in {metric}. "
        f"{fake.text(max_nb_chars=100)} "
        f'"{fake.catch_phrase()}," said the CEO of {client_name}. '
        f"Key Takeaways: {fake.catch_phrase()}. {fake.bs().title()}."
    )
    
    if use_gemini and gemini_model is not None:
        prompt = f"""Write a professional case study for {company_name}.
Client: {client_name}
Result: {improvement}% improvement in {metric}

Write a brief case study with:
1. Challenge faced by the client
2. Solution provided by {company_name}
3. Results achieved ({improvement}% improvement in {metric})
4. A testimonial quote
5. Key takeaways

Keep it professional and concise (3-4 sentences)."""
        
        is_seed = client_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = fallback_content
    
    return {
        'document_type': 'case_study',
        'title': f'Case Study: {client_name} Success with {company_name}',
        'company_name': company_name,
        'content': content,
        'client_name': client_name,
        'improvement': improvement,
        'metric': metric
    }


def generate_internal_memo(client_data, use_gemini=False):
    """Generates an internal company memo.
    
    Args:
        client_data (dict): Company information
        use_gemini (bool): If True, use Gemini AI for content generation
    """
    company_name = client_data.get('company_name', 'Company')
    
    topics = [
        "Quarterly Performance Update",
        "New Policy Implementation",
        "Team Restructuring Announcement",
        "Strategic Initiative Launch",
        "Process Improvement Notice"
    ]
    
    topic = random.choice(topics)
    date_str = fake.date_this_year().strftime('%B %d, %Y')
    
    fallback_content = (
        f"INTERNAL MEMORANDUM - {company_name}. "
        f"Date: {date_str}. "
        f"To: All Staff. From: Management. "
        f"Subject: {topic}. "
        f"{fake.text(max_nb_chars=120)} "
        f"Key points: {fake.catch_phrase()}. {fake.bs().title()}. {fake.catch_phrase()}. "
        f"{fake.text(max_nb_chars=80)} "
        f"Please contact your department head with any questions."
    )
    
    if use_gemini and gemini_model is not None:
        prompt = f"""Write a professional internal memo for {company_name}.
Date: {date_str}
To: All Staff
From: Management
Subject: {topic}

Write a brief memo with key points and action items. Keep it professional and concise (3-4 sentences)."""
        
        is_seed = client_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = fallback_content
    
    return {
        'document_type': 'internal_memo',
        'title': f'{company_name} - Internal Memo: {topic}',
        'company_name': company_name,
        'content': content
    }


def generate_partnership_document(company_data_1, company_data_2):
    """Generates a partnership or collaboration document between two companies."""
    company_1 = company_data_1.get('company_name', 'Company A')
    company_2 = company_data_2.get('company_name', 'Company B')
    
    partnership_types = [
        "Strategic Alliance Agreement",
        "Joint Venture Proposal",
        "Co-Marketing Agreement",
        "Technology Partnership",
        "Distribution Agreement"
    ]
    
    partnership_type = random.choice(partnership_types)
    
    return {
        'document_type': 'partnership_document',
        'title': f'{partnership_type}: {company_1} & {company_2}',
        'company_name': f'{company_1}, {company_2}',
        'companies': [company_1, company_2],
        'content': (
            f"{partnership_type} between {company_1} and {company_2}. "
            f"Date: {fake.date_this_year().strftime('%B %d, %Y')}. "
            f"Purpose: {fake.catch_phrase()}. "
            f"{company_1} and {company_2} hereby agree to collaborate on {fake.bs().lower()}. "
            f"Terms: {fake.text(max_nb_chars=100)} "
            f"Expected outcomes: {fake.catch_phrase()}. {fake.bs().title()}. "
            f"Duration: {random.randint(1, 5)} years. "
            f"{fake.text(max_nb_chars=80)} "
            f"Both parties commit to {fake.catch_phrase().lower()} for mutual benefit."
        ),
        'partnership_type': partnership_type
    }


def generate_shareholder_report(company_data, use_gemini=False):
    """Generates a shareholder report (can be shared among investors)."""
    company_name = company_data.get('company_name', 'Company')
    year = random.randint(2022, 2025)
    
    shareholders = [fake.name() for _ in range(random.randint(3, 7))]
    
    # Prepare fallback content with Faker
    fallback_content = (
        f"Annual Shareholder Report - {company_name} ({year}). "
        f"Dear Valued Shareholders, "
        f"{fake.text(max_nb_chars=100)} "
        f"Financial Performance: Revenue growth of {random.uniform(5, 30):.1f}%. "
        f"Earnings per share: ${random.uniform(0.50, 5.00):.2f}. "
        f"Strategic Initiatives: {fake.catch_phrase()}. {fake.bs().title()}. "
        f"{fake.text(max_nb_chars=120)} "
        f"Board of Directors: {', '.join(shareholders[:3])}. "
        f"Looking ahead: {fake.catch_phrase()}. {fake.text(max_nb_chars=80)}"
    )
    
    # Generate content with Gemini or fallback to Faker
    if use_gemini:
        industry = company_data.get('industry', 'Technology')
        revenue_growth = random.uniform(5, 30)
        eps = random.uniform(0.50, 5.00)
        prompt = f"""Generate a comprehensive annual shareholder report for {company_name}, a {industry} company.

Include the following sections:
1. Letter to Shareholders - Address from CEO/Board thanking shareholders and summarizing the year
2. Financial Highlights - Revenue growth of {revenue_growth:.1f}%, earnings per share ${eps:.2f}, key financial metrics
3. Strategic Initiatives - Major projects, expansions, or strategic directions undertaken this year
4. Market Position - Competitive landscape, market share, industry trends
5. Board of Directors - Current board members: {', '.join(shareholders[:3])}
6. Future Outlook - Goals, expectations, and planned initiatives for the coming year
7. Risk Factors - Key risks and how they're being managed
8. Shareholder Value - Dividend policy, stock performance, capital allocation strategy

Make it professional, detailed, and optimistic while remaining realistic. Target length: 400-500 words."""
        
        is_seed = company_data.get('is_seed', False)
        content = _generate_with_gemini(prompt, fallback_content, is_seed_company=is_seed)
    else:
        content = fallback_content
    
    return {
        'document_type': 'shareholder_report',
        'title': f'{company_name} - Annual Shareholder Report {year}',
        'company_name': company_name,
        'year': year,
        'content': content,
        'shareholders': shareholders
    }


def generate_all_documents_for_company(client_data, num_docs_range=(5, 10), use_gemini=False):
    """
    Generates multiple diverse documents for a single company.
    
    Args:
        client_data (dict): Company information
        num_docs_range (tuple): Min and max number of documents to generate
        use_gemini (bool): If True, use Gemini AI for content generation
        
    Returns:
        list: List of document dictionaries, or None if Gemini fails for a seed company
    """
    is_seed_company = client_data.get('is_seed', False)
    num_docs = random.randint(num_docs_range[0], num_docs_range[1])
    documents = []
    
    try:
        # Always include at least one of each core type
        # Generate TWO versions for product and services brochures (one for image, one for PDF)
        if use_gemini:
            print(f"      📄 Generating product brochure...")
        product_brochure_img = generate_product_brochure(client_data, use_gemini=use_gemini, variation='A')
        product_brochure_pdf = generate_product_brochure(client_data, use_gemini=use_gemini, variation='B')
        # Combine both versions into one document with content_image and content_pdf
        product_brochure = product_brochure_img.copy()
        product_brochure['content_image'] = product_brochure_img['content']
        product_brochure['content_pdf'] = product_brochure_pdf['content']
        documents.append(product_brochure)
        
        if use_gemini:
            print(f"      📄 Generating services brochure...")
        services_brochure_img = generate_services_brochure(client_data, use_gemini=use_gemini, variation='A')
        services_brochure_pdf = generate_services_brochure(client_data, use_gemini=use_gemini, variation='B')
        # Combine both versions
        services_brochure = services_brochure_img.copy()
        services_brochure['content_image'] = services_brochure_img['content']
        services_brochure['content_pdf'] = services_brochure_pdf['content']
        documents.append(services_brochure)
        
        if use_gemini:
            print(f"      📄 Generating financial report...")
        documents.append(generate_financial_report(client_data, use_gemini=use_gemini))
        
        if use_gemini:
            print(f"      📄 Generating press release...")
        documents.append(generate_press_release(client_data, use_gemini=use_gemini))
        
        # Add additional random documents
        # For real companies: exclude internal memos (private) and shareholder reports (hard to verify)
        # Focus on documents with easily verifiable, realistic content
        if is_seed_company:
            available_generators = [
                ('product_brochure', lambda cd: generate_product_brochure(cd, use_gemini=use_gemini)),
                ('services_brochure', lambda cd: generate_services_brochure(cd, use_gemini=use_gemini)),
                ('financial_report', lambda cd: generate_financial_report(cd, use_gemini=use_gemini)),
                ('press_release', lambda cd: generate_press_release(cd, use_gemini=use_gemini)),
                ('advertisement', lambda cd: generate_advertisement(cd, use_gemini=use_gemini)),
                ('case_study', lambda cd: generate_case_study(cd, use_gemini=use_gemini)),
                # SKIP shareholder_report for real companies - harder to make 100% accurate
            ]
        else:
            # For synthetic companies: include all document types
            available_generators = [
                ('product_brochure', lambda cd: generate_product_brochure(cd, use_gemini=use_gemini)),
                ('services_brochure', lambda cd: generate_services_brochure(cd, use_gemini=use_gemini)),
                ('financial_report', lambda cd: generate_financial_report(cd, use_gemini=use_gemini)),
                ('press_release', lambda cd: generate_press_release(cd, use_gemini=use_gemini)),
                ('advertisement', lambda cd: generate_advertisement(cd, use_gemini=use_gemini)),
                ('case_study', lambda cd: generate_case_study(cd, use_gemini=use_gemini)),
                ('internal_memo', lambda cd: generate_internal_memo(cd, use_gemini=use_gemini)),
                ('shareholder_report', lambda cd: generate_shareholder_report(cd, use_gemini=use_gemini))
            ]
        
        for i in range(num_docs - 4):
            doc_type, generator = random.choice(available_generators)
            if use_gemini:
                print(f"      📄 Generating {doc_type.replace('_', ' ')}... ({i+5}/{num_docs})")
            documents.append(generator(client_data))
        
        return documents
        
    except Exception as e:
        # If this is a seed company and Gemini failed, mark it and return None
        if is_seed_company:
            print(f"❌ Gemini failed for real company '{client_data.get('company_name')}': {e}")
            mark_gemini_failed()  # Signal to skip remaining seed companies
            return None  # Don't mix Gemini + Faker for real companies
        else:
            # For synthetic companies, this shouldn't happen (fallback is built in)
            print(f"⚠️ Unexpected error for synthetic company: {e}")
            raise  # Re-raise to show there's a problem


def generate_shared_documents(all_companies_data, num_partnerships=10):
    """
    Generates documents that are shared between companies (partnerships, etc.).
    
    Args:
        all_companies_data (list): List of all company data dictionaries
        num_partnerships (int): Number of partnership documents to create
        
    Returns:
        list: List of shared document dictionaries
    """
    shared_docs = []
    
    for _ in range(num_partnerships):
        if len(all_companies_data) >= 2:
            company_1, company_2 = random.sample(all_companies_data, 2)
            shared_docs.append(generate_partnership_document(company_1, company_2))
    
    return shared_docs

