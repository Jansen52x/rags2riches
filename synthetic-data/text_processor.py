
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# Initialize NLTK resources
def setup_nltk():
  try:
    nltk.data.find('tokenizers/punkt')
  except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
  
  try:
    nltk.data.find('corpora/stopwords')
  except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)


# Set up stop words
setup_nltk()
stop_words = set(stopwords.words('english'))


def combine_and_clean_text(row):
 
  client_info = row['client_data']
  industry_info = row['industry_overview']

  # Extract relevant text from the client_data dictionary
  client_text = (
    f"Company: {client_info.get('company_name', '')}. "
    f"Industry: {client_info.get('industry', '')}. "
    f"Contact: {client_info.get('contact_person', '')}. "
    f"Email: {client_info.get('contact_email', '')}. "
    f"Description: {client_info.get('company_description', '')}"
  )

  # Combine client info and industry overview
  combined_text = f"{client_text} Industry Overview: {industry_info}"

  # Clean the text
  # Remove special characters and punctuation
  combined_text = re.sub(f'[{re.escape(string.punctuation)}]', '', combined_text)
  # Convert to lowercase
  combined_text = combined_text.lower()
  # Remove extra whitespace
  combined_text = re.sub(r'\s+', ' ', combined_text).strip()

  return combined_text


def tokenize_and_remove_stopwords_robust(text):

  if not isinstance(text, str):
    return [] # Return empty list for non-string inputs

  tokenizer = RegexpTokenizer(r'\w+') # Tokenize into words, ignoring punctuation
  tokens = tokenizer.tokenize(text)

  filtered_tokens = [word for word in tokens if word not in stop_words]
  return filtered_tokens


def process_dataframe(df):

  # First, combine and clean the text (returns string)
  df['document_text'] = df.apply(combine_and_clean_text, axis=1)
  
  # Then tokenize and remove stopwords (returns list of tokens)
  df['document_text'] = df['document_text'].apply(tokenize_and_remove_stopwords_robust)
  
  return df
