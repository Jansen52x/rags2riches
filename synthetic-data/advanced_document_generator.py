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
"""
from faker import Faker
import random

fake = Faker()


def generate_product_brochure(client_data):
    """Generates a product-focused brochure."""
    company_name = client_data.get('company_name', 'Company')
    industry = client_data.get('industry', 'Industry')
    
    products = [
        f"{fake.word().title()} {fake.word().title()} System",
        f"{fake.color_name().title()} {fake.word().title()} Platform",
        f"Premium {fake.word().title()} Suite",
        f"Enterprise {fake.word().title()} Solution"
    ]
    
    product_name = random.choice(products)
    
    return {
        'document_type': 'product_brochure',
        'title': f'{product_name} - Product Brochure',
        'company_name': company_name,
        'content': (
            f"Introducing {product_name} by {company_name}. "
            f"{fake.catch_phrase()}. "
            f"Our innovative solution in the {industry} sector delivers {fake.bs().lower()}. "
            f"Key features include: {fake.catch_phrase().lower()}, {fake.bs().lower()}, "
            f"and {fake.catch_phrase().lower()}. "
            f"{fake.text(max_nb_chars=150)} "
            f"Contact us today to learn more about how {product_name} can transform your business."
        ),
        'product_name': product_name
    }


def generate_services_brochure(client_data):
    """Generates a services-focused brochure."""
    company_name = client_data.get('company_name', 'Company')
    industry = client_data.get('industry', 'Industry')
    
    services = [
        f"{fake.word().title()} Consulting",
        f"{fake.word().title()} Management Services",
        f"Professional {fake.word().title()} Support",
        f"{fake.word().title()} Implementation"
    ]
    
    return {
        'document_type': 'services_brochure',
        'title': f'{company_name} - Services Overview',
        'company_name': company_name,
        'content': (
            f"{company_name} offers comprehensive services in {industry}. "
            f"Our service portfolio includes: {', '.join(random.sample(services, 3))}. "
            f"{fake.catch_phrase()}. "
            f"With over {random.randint(5, 20)} years of experience, we provide {fake.bs().lower()}. "
            f"{fake.text(max_nb_chars=120)} "
            f"Our team of experts ensures {fake.catch_phrase().lower()} for all our clients."
        )
    }


def generate_financial_report(client_data, quarter=None, year=None):
    """Generates a quarterly or annual financial report."""
    company_name = client_data.get('company_name', 'Company')
    
    if year is None:
        year = random.randint(2022, 2025)
    
    if quarter is None:
        quarter = random.choice(['Q1', 'Q2', 'Q3', 'Q4', 'Annual'])
    
    revenue = random.randint(500000, 50000000)
    growth = random.uniform(-5, 25)
    profit_margin = random.uniform(5, 30)
    
    return {
        'document_type': 'financial_report',
        'title': f'{company_name} - {quarter} {year} Financial Report',
        'company_name': company_name,
        'quarter': quarter,
        'year': year,
        'content': (
            f"{company_name} Financial Report - {quarter} {year}. "
            f"Revenue: ${revenue:,.2f}. Year-over-year growth: {growth:.1f}%. "
            f"Operating profit margin: {profit_margin:.1f}%. "
            f"Executive Summary: {fake.text(max_nb_chars=100)} "
            f"Key Highlights: {fake.catch_phrase()}. {fake.bs().title()}. "
            f"Market conditions: {fake.text(max_nb_chars=80)} "
            f"Future outlook: {fake.catch_phrase()}. {fake.bs().title()}."
        ),
        'revenue': revenue,
        'growth_rate': growth,
        'profit_margin': profit_margin
    }


def generate_press_release(client_data):
    """Generates a press release."""
    company_name = client_data.get('company_name', 'Company')
    contact_person = client_data.get('contact_person', 'Contact')
    contact_email = client_data.get('contact_email', 'email@example.com')
    
    topics = [
        f"Announces Strategic Partnership",
        f"Launches Revolutionary New Product",
        f"Expands Operations",
        f"Achieves Major Milestone",
        f"Wins Industry Award",
        f"Appoints New Leadership"
    ]
    
    topic = random.choice(topics)
    
    return {
        'document_type': 'press_release',
        'title': f'{company_name} {topic}',
        'company_name': company_name,
        'content': (
            f"FOR IMMEDIATE RELEASE. "
            f"{company_name} {topic}. "
            f"{fake.city()}, {fake.date_this_year().strftime('%B %d, %Y')} - "
            f"{company_name} today announced {fake.catch_phrase().lower()}. "
            f"{fake.text(max_nb_chars=150)} "
            f'"{fake.catch_phrase()}," said {contact_person}, spokesperson for {company_name}. '
            f"{fake.text(max_nb_chars=100)} "
            f"For more information, contact {contact_email}."
        ),
        'contact_person': contact_person,
        'contact_email': contact_email
    }


def generate_advertisement(client_data):
    """Generates an advertisement."""
    company_name = client_data.get('company_name', 'Company')
    industry = client_data.get('industry', 'Industry')
    
    headlines = [
        f"Transform Your Business Today",
        f"The Future of {industry.split()[0].title()} is Here",
        f"Experience Excellence",
        f"Innovation You Can Trust",
        f"Leading the Way in {industry.split()[0].title()}"
    ]
    
    return {
        'document_type': 'advertisement',
        'title': f'{company_name} - Advertisement',
        'company_name': company_name,
        'content': (
            f"{random.choice(headlines)}! "
            f"{company_name} - {fake.catch_phrase()}. "
            f"Discover how we're {fake.bs().lower()}. "
            f"Why choose us? {fake.catch_phrase()}. {fake.catch_phrase()}. {fake.catch_phrase()}. "
            f"{fake.text(max_nb_chars=80)} "
            f"Don't wait! Contact us today and join the revolution in {industry}."
        )
    }


def generate_case_study(client_data):
    """Generates a case study."""
    company_name = client_data.get('company_name', 'Company')
    
    client_names = [fake.company() for _ in range(3)]
    client_name = random.choice(client_names)
    
    improvement = random.randint(15, 85)
    metric = random.choice(['efficiency', 'revenue', 'productivity', 'customer satisfaction', 'market share'])
    
    return {
        'document_type': 'case_study',
        'title': f'Case Study: {client_name} Success with {company_name}',
        'company_name': company_name,
        'content': (
            f"Case Study: How {client_name} Achieved Success with {company_name}. "
            f"Challenge: {fake.text(max_nb_chars=80)} "
            f"Solution: {company_name} implemented {fake.catch_phrase().lower()} to help {client_name} {fake.bs().lower()}. "
            f"Results: {improvement}% improvement in {metric}. "
            f"{fake.text(max_nb_chars=100)} "
            f'"{fake.catch_phrase()}," said the CEO of {client_name}. '
            f"Key Takeaways: {fake.catch_phrase()}. {fake.bs().title()}."
        ),
        'client_name': client_name,
        'improvement': improvement,
        'metric': metric
    }


def generate_internal_memo(client_data):
    """Generates an internal company memo."""
    company_name = client_data.get('company_name', 'Company')
    
    topics = [
        "Quarterly Performance Update",
        "New Policy Implementation",
        "Team Restructuring Announcement",
        "Strategic Initiative Launch",
        "Process Improvement Notice"
    ]
    
    return {
        'document_type': 'internal_memo',
        'title': f'{company_name} - Internal Memo: {random.choice(topics)}',
        'company_name': company_name,
        'content': (
            f"INTERNAL MEMORANDUM - {company_name}. "
            f"Date: {fake.date_this_year().strftime('%B %d, %Y')}. "
            f"To: All Staff. From: Management. "
            f"Subject: {random.choice(topics)}. "
            f"{fake.text(max_nb_chars=120)} "
            f"Key points: {fake.catch_phrase()}. {fake.bs().title()}. {fake.catch_phrase()}. "
            f"{fake.text(max_nb_chars=80)} "
            f"Please contact your department head with any questions."
        )
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


def generate_shareholder_report(company_data):
    """Generates a shareholder report (can be shared among investors)."""
    company_name = company_data.get('company_name', 'Company')
    year = random.randint(2022, 2025)
    
    shareholders = [fake.name() for _ in range(random.randint(3, 7))]
    
    return {
        'document_type': 'shareholder_report',
        'title': f'{company_name} - Annual Shareholder Report {year}',
        'company_name': company_name,
        'year': year,
        'content': (
            f"Annual Shareholder Report - {company_name} ({year}). "
            f"Dear Valued Shareholders, "
            f"{fake.text(max_nb_chars=100)} "
            f"Financial Performance: Revenue growth of {random.uniform(5, 30):.1f}%. "
            f"Earnings per share: ${random.uniform(0.50, 5.00):.2f}. "
            f"Strategic Initiatives: {fake.catch_phrase()}. {fake.bs().title()}. "
            f"{fake.text(max_nb_chars=120)} "
            f"Board of Directors: {', '.join(shareholders[:3])}. "
            f"Looking ahead: {fake.catch_phrase()}. {fake.text(max_nb_chars=80)}"
        ),
        'shareholders': shareholders
    }


def generate_all_documents_for_company(client_data, num_docs_range=(5, 10)):
    """
    Generates multiple diverse documents for a single company.
    
    Args:
        client_data (dict): Company information
        num_docs_range (tuple): Min and max number of documents to generate
        
    Returns:
        list: List of document dictionaries
    """
    num_docs = random.randint(num_docs_range[0], num_docs_range[1])
    documents = []
    
    # Always include at least one of each core type
    documents.append(generate_product_brochure(client_data))
    documents.append(generate_services_brochure(client_data))
    documents.append(generate_financial_report(client_data))
    documents.append(generate_press_release(client_data))
    
    # Add additional random documents
    available_generators = [
        generate_product_brochure,
        generate_services_brochure,
        generate_financial_report,
        generate_press_release,
        generate_advertisement,
        generate_case_study,
        generate_internal_memo,
        generate_shareholder_report
    ]
    
    for _ in range(num_docs - 4):
        generator = random.choice(available_generators)
        documents.append(generator(client_data))
    
    return documents


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
