#11111111
# Module or library install command (run this in terminal before running the script)
# pip install gensim scipy

# Import required libraries
import gensim.downloader as api  # For downloading pre-trained word vectors
from scipy.spatial.distance import cosine  # For calculating cosine similarity

# Load pre-trained Word2Vec model (Google News, 300 dimensions)
print("Loading Word2Vec model...")
model = api.load("word2vec-google-news-300")
print("Model loaded successfully.\n")

# Get and print the first 10 dimensions of the word vector for 'king'
vector = model['king']
print("First 10 dimensions of 'king' vector:")
print(vector[:10], "\n")

# Print top 10 most similar words to 'king'
print("Top 10 words most similar to 'king':")
for word, similarity in model.most_similar('king'):
    print(f"{word}: {similarity:.4f}")
print()

# Perform word analogy: king - man + woman ≈ queen
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("Analogy - 'king' - 'man' + 'woman' ≈ ?")
print(f"Result: {result[0][0]} (Similarity: {result[0][1]:.4f})\n")

# Analogy: paris + italy - france ≈ rome
print("Analogy - 'paris' + 'italy' - 'france' ≈ ?")
for word, similarity in model.most_similar(positive=['paris', 'italy'], negative=['france']):
    print(f"{word}: {similarity:.4f}")
print()

# Analogy: walking + swimming - walk ≈ swim
print("Analogy - 'walking' + 'swimming' - 'walk' ≈ ?")
for word, similarity in model.most_similar(positive=['walking', 'swimming'], negative=['walk']):
    print(f"{word}: {similarity:.4f}")
print()

# Calculate cosine similarity between 'king' and 'queen'
similarity = 1 - cosine(model['king'], model['queen'])
print(f"Cosine similarity between 'king' and 'queen': {similarity:.4f}")









#22222222
# Module or library install command (run this in terminal before running the script)
# pip install gensim matplotlib scikit-learn

import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load model
model = api.load("word2vec-google-news-300")

# Select 10 domain-specific words (technology domain)
words = ['computer', 'internet', 'software', 'hardware', 'keyboard', 'mouse', 'server', 'network', 'programming', 'database']
vectors = [model[word] for word in words]

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Generate 5 semantically similar words for a given input
input_word = 'computer'
similar_words = model.most_similar(input_word, topn=5)

# Print the similar words to terminal
print(f"Top 5 words similar to '{input_word}':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

# Plot the word embeddings
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]))
plt.title("PCA Visualization of Technology Word Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

# Show the plot
plt.show()
















#Program 3. Train a custom Word2Vec model on a small dataset. Train embeddings on a domain 
specific corpus (e.g., legal, medical) and analyze how embeddings capture domain-specific 
semantics. 
import gensim 
from gensim.models import Word2Vec 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
import numpy as np 
# Sample domain-specific corpus (Medical text examples) 
corpus = [ 
"The patient received treatment for muscle fatigue.", 
"The doctor confirmed the diagnosis and recommended therapy.", 
"A new vaccine is recommended for disease prevention.", 
"The procedure for the surgery was successful.", 
"Monitoring the patient's recovery is crucial." 
] 
# Tokenize the corpus 
sentences = [word_tokenize(sentence.lower()) for sentence in corpus] 
# Train Word2Vec model 
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4) 
# Save and load model 
model.save("medical_word2vec.model") 
model = Word2Vec.load("medical_word2vec.model") 
# Function to display similar words 
def display_similar_words(word): 
if word in model.wv: 
similar_words = model.wv.most_similar(word, topn=10) 
print(f"Words similar to '{word}':") 
for w, score in similar_words: 
print(f"  {w} ({score:.2f})") 
return [word] + [w for w, _ in similar_words] 
else: 
print(f"'{word}' not in vocabulary") 
return [] 
# Function to visualize embeddings using PCA 
def plot_word_embeddings(words): 
word_vectors = np.array([model.wv[w] for w in words if w in model.wv]) 
pca = PCA(n_components=2) 
reduced_vectors = pca.fit_transform(word_vectors) 
plt.figure(figsize=(10, 8)) 
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color='blue') 
for i, word in enumerate(words): 
plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=10, color='black') 
plt.title("Word Embeddings Visualization (Medical Domain)") 
plt.xlabel("Dimension 1") 
plt.ylabel("Dimension 2") 
plt.grid(True) 
plt.show() 
# Analyze specific words 
words_treatment = display_similar_words("treatment") 
words_vaccine = display_similar_words("vaccine") 
# Plot graph for visualization 
plot_word_embeddings(words_treatment + words_vaccine)



















#4.Use word embeddings to improve prompts for Generative AI model. Retrieve similar 
words using word embeddings. Use the similar words to enrich a GenAI prompt. Use the 
AI model to generate responses for the original and enriched prompts. Compare the 
outputs in terms of detail and relevance. 
import nltk 
import string 
from nltk.tokenize import word_tokenize 
from transformers import pipeline 
import gensim.downloader as api 
# Download necessary NLTK data 
nltk.download('punkt') 
# Load pre-trained word vectors safely 
try: 
print("Loading pre-trained word vectors...") 
word_vectors = api.load("glove-wiki-gigaword-100")  # Correct way to load GloVe in gensim 
except Exception as e: 
print(f"Error loading word vectors: {e}") 
word_vectors = None 
# Function to replace a keyword with its most similar word 
def replace_keyword(prompt, keyword, word_vectors): 
words = word_tokenize(prompt) 
enriched_words = [] 
for word in words: 
cleaned_word = word.lower().strip(string.punctuation) 
if cleaned_word == keyword.lower() and word_vectors and cleaned_word in word_vectors: 
try: 
replacement = word_vectors.most_similar(cleaned_word, topn=1)[0][0] 
print(f"Replacing '{keyword}' → '{replacement}'") 
enriched_words.append(replacement) 
continue 
except KeyError: 
pass 
enriched_words.append(word) 
return " ".join(enriched_words) 
# Load GPT-2 model safely 
try: 
print("Loading GPT-2 model...") 
generator = pipeline("text-generation", model="gpt2") 
except Exception as e: 
print(f"Error loading GPT-2: {e}") 
generator = None 
# Function to generate response 
def generate_response(prompt): 
if generator: 
try: 
return generator(prompt, max_length=100, num_return_sequences=1, 
truncation=True)[0]['generated_text'] 
except Exception as e: 
return f"Error generating response: {e}" 
return "No response generated (model loading failed)." 
# Example usage 
original_prompt = "Who is king." 
key_term = "king" 
enriched_prompt = replace_keyword(original_prompt, key_term, word_vectors) if word_vectors 
else original_prompt 
print("\n🔹 Original Prompt:", original_prompt) 
print("🔹 Enriched Prompt:", enriched_prompt) 
print("\n🔹 Generating response for the original prompt...") 
original_response = generate_response(original_prompt) 
print("\nOriginal Prompt Response:\n", original_response) 
print("\n🔹 Generating response for the enriched prompt...") 
enriched_response = generate_response(enriched_prompt) 
print("\nEnriched Prompt Response:\n", enriched_response) 
# Comparison Metrics 
print("\n🔹 Comparison of Responses:") 
print(f"Original Prompt Response Length: {len(original_response)}") 
print(f"Enriched Prompt Response Length: {len(enriched_response)} 















5.Use word embeddings to create meaningful sentences for creative tasks. Retrieve similar 
words for a seed word. Create a sentence or story using these words as a starting point. 
Write a program that: Takes a seed word. Generates similar words. Constructs a short 
paragraph using these words. 
import nltk 
import string 
import gensim.downloader as api 
from nltk.tokenize import word_tokenize 
from transformers import pipeline 
import random 
nltk.download('punkt') 
# Load pre-trained word vectors 
try: 
word_vectors = api.load("glove-wiki-gigaword-100")  # Load GloVe embeddings 
print("Word vectors loaded successfully!") 
except Exception as e: 
print(f"Error loading word vectors: {e}") 
word_vectors = None 
# Function to replace a keyword with its most similar word 
def replace_keyword(prompt, keyword, word_vectors): 
words = word_tokenize(prompt) 
enriched_words = [] 
for word in words: 
cleaned_word = word.lower().strip(string.punctuation) 
if cleaned_word == keyword.lower() and word_vectors: 
try: 
replacement = word_vectors.most_similar(cleaned_word, topn=1)[0][0] 
enriched_words.append(replacement) 
continue 
except KeyError: 
pass 
enriched_words.append(word) 
return " ".join(enriched_words) 
# Load GPT-2 model 
generator = pipeline("text-generation", model="gpt2") 
def generate_response(prompt): 
return generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text'] 
def generate_paragraph(seed_word): 
"""Construct a paragraph using similar words.""" 
similar_words = word_vectors.most_similar(seed_word, topn=5) if word_vectors else [] 
if not similar_words: 
return "Could not generate a paragraph. Try another seed word." 
paragraph_templates = [ 
f"In the land of {seed_word}, {similar_words[4][0]} was a common sight.", 
f"People often associate {seed_word} with {similar_words[2][0]} and 
{similar_words[3][0]}.", 
f"A story about {seed_word} would be incomplete without {similar_words[1][0]} and 
{similar_words[3][0]}.", 
] 
paragraph = " ".join(random.sample(paragraph_templates, len(paragraph_templates))) 
return paragraph 
# Example usage 
seed_word = input("Enter a seed word: ") 
print("\nGenerated Paragraph:\n") 
print(generate_paragraph(seed_word))
















#Program 06
!pip install transformers

from transformers import pipeline

print(" Loading Sentiment Analysis Model...")
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
 """
 Analyze the sentiment of a given text input.

 Args:
 text (str): Input sentence or paragraph.

 Returns:
 dict: Sentiment label and confidence score.
 """
 result = sentiment_analyzer(text)[0] 
 label = result['label'] 
 score = result['score'] 

 print(f"\n Input Text: {text}")
 print(f" Sentiment: {label} (Confidence: {score:.4f})\n")
 return result

customer_reviews = [
 "The product is amazing! I love it so much.",
 "I'm very disappointed. The service was terrible.",
 "It was an average experience, nothing special.",
 "Absolutely fantastic quality! Highly recommended.",
 "Not great, but not the worst either."
]

print("\n Customer Sentiment Analysis Results:")
for review in customer_reviews:
  analyze_sentiment(review)
 












 #program 07

 #program 7


from transformers import pipeline
# Step 1: Load the model
print("Loading Summarization Model (BART)...")
summarizer = pipeline("summarization")
print("Device set to use: CPU")
# Step 2: Original text to summarize
original_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on
creating intelligent machines capable of mimicking human cognitive functions such as
learning, problem-solving, and decision-making. In recent years, AI has significantly
impacted various industries, including healthcare, finance, education, and entertainment.
AI-powered applications, such as chatbots, self-driving cars, and recommendation
systems, have transformed the way we interact with technology. Machine learning and
deep learning, subsets of AI, enable systems to learn from data and improve over time
without explicit programming. However, AI also poses ethical challenges, such as bias in
decision-making and concerns over job displacement. As AI technology continues to
advance, it is crucial to balance innovation with ethical considerations to ensure its
responsible development and deployment.
"""
print("\nOriginal Text:\n")
print(original_text)
# Step 3: Summarize the text
summary = summarizer(original_text, max_length=100, min_length=30, do_sample=False)
# Step 4: Display the summary
print("\n--- Summary ---\n")
print(summary[0]['summary_text'])


















#PROGRAM 8

!pip install langchain cohere langchain-community --quiet

import cohere
import getpass
from langchain import PromptTemplate
from langchain.llms import Cohere

file_path = "Teaching.txt" 
try:
   with open(file_path, "r", encoding="utf-8") as file:
     text_content = file.read()
   print("File loaded successfully!")
except Exception as e:
   print(" Error loading file:", str(e))
   text_content = " hey hi i am persuing be in artificail intelligence and machine learning" 

COHERE_API_KEY = getpass.getpass(" Enter your Cohere API Key: ")

cohere_llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command")

template = """
You are an AI assistant helping to summarize and analyze a text document.
Here is the document content:
{text}
Summary: Provide a concise summary of the document.
Key Takeaways: List 3 important points from the text.
Sentiment Analysis: Determine if the sentiment of the document is Positive, Negative, or Neutral.
"""
prompt_template = PromptTemplate(input_variables=["text"], template=template)

if text_content.strip():
 formatted_prompt = prompt_template.format(text=text_content)
 response = cohere_llm.predict(formatted_prompt)
 
 print("\n **Formatted Output** \n")
 print(response)
else:
 print("No text to analyze.")
















#program 09
# Install required dependencies (Run this in Jupyter or terminal before executing the script)
!pip install wikipedia-api ipywidgets pydantic --quiet

# Import necessary libraries
import wikipediaapi
import ipywidgets as widgets
from IPython.display import display
from pydantic import BaseModel, Field
import re

# Define InstitutionDetails model to store extracted information
class InstitutionDetails(BaseModel):
    founder: str = Field(default="Not Available")
    founded: str = Field(default="Not Available")
    branches: str = Field(default="Not Available")
    employees: str = Field(default="Not Available")    
    summary: str

# Function to fetch institution details from Wikipedia
def fetch_institution_details(institution_name: str) -> InstitutionDetails:

    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='InstitutionFetcherBot/1.0 (kai.rohistudent@gmail.com)'
    )

    page = wiki_wiki.page(institution_name)

    if not page.exists():
        return InstitutionDetails(summary="Institution not found on Wikipedia.")

    content = page.text

    founder = re.search(r"[Ff]ounder[s]?:?\s*(.*)", content)
    founded = re.search(r"[Ff]ounded:?[\s](.?\d{4})", content)
    employees = re.search(r"([Ee]mployee[s]?|[Ff]aculty):?\s*([\d,]+)", content)
    branches = re.findall(r"[Cc]ampus(?:es)?(?: include)?(?:s)?:?\s*(.*)", content)

    sentences = content.split('. ')
    summary = '. '.join(sentences[:4]).strip() + "."

    return InstitutionDetails(
        founder=founder.group(1).split('.')[0] if founder else "Not Available",
        founded=founded.group(1).split('.')[0] if founded else "Not Available",
        employees=employees.group(2) if employees else "Not Available",
        branches=branches[0] if branches else "Not Available",
        summary=summary
    )

# Create interactive widgets
input_box = widgets.Text(
    description="Institution:",
    placeholder="Enter institution name",
    layout=widgets.Layout(width='400px')
)

fetch_button = widgets.Button(
    description="Fetch Details",
    button_style="success"
)

output_box = widgets.Output()

# Define button click event function
def on_fetch_button_clicked(b):
    with output_box:
        output_box.clear_output()
        institution_name = input_box.value.strip()

        if not institution_name:
            print("❌ Please enter a valid institution name!")
            return

        details = fetch_institution_details(institution_name)

        print(f"\n📌 Institution: {institution_name}\n")
        print(f"🔹 Founder: {details.founder}")
        print(f"🔹 Founded: {details.founded}")
        print(f"🔹 Branches: {details.branches}")
        print(f"🔹 Employees: {details.employees}\n")
        print(f"📝 Summary:\n{details.summary}")

# Link button click event to function
fetch_button.on_click(on_fetch_button_clicked)

# Ensure widgets display correctly
display(input_box)
display(fetch_button)
display(output_box)










#program 10
!pip install langchain cohere wikipedia-api pydantic ipywidgets

from langchain import PromptTemplate
from langchain.llms import Cohere
from pydantic import BaseModel
from typing import Optional
import wikipediaapi
from IPython.display import display
import ipywidgets as widgets
import getpass


COHERE_API_KEY = getpass.getpass('Enter your Cohere API Key: ')
cohere_llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command")


def fetch_ipc_summary() -> str:
    user_agent = "IPCChatbot/1.0 (contact: myemail@example.com)" 
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')
    page = wiki_wiki.page("Indian Penal Code")
    if not page.exists():
        raise ValueError("The Indian Penal Code page does not exist on Wikipedia.")
    return page.text[:5000]  


try:
    ipc_content = fetch_ipc_summary()
except Exception as e:
    ipc_content = "Could not fetch IPC content from Wikipedia. Please check your connection."
    print(f"Error: {e}")


class IPCResponse(BaseModel):
    section: Optional[str]
    explanation: Optional[str]


prompt_template = PromptTemplate(
    input_variables=["ipc_content", "question"],
    template="""
    You are a legal assistant chatbot specialized in the Indian Penal Code (IPC).
    Refer to the following IPC document content to answer the user's query:
    {ipc_content}
    User Question: {question}
    Provide a detailed answer, mentioning the relevant section if applicable.
    """
)


def get_ipc_response(question: str) -> IPCResponse:
    try:
        formatted_prompt = prompt_template.format(ipc_content=ipc_content, question=question)
        response = cohere_llm.predict(formatted_prompt)
      
        if "Section" in response:
            section = response.split('Section')[1].split(':')[0].strip()
            explanation = response.split(':', 1)[-1].strip()
        else:
            section = None
            explanation = response.strip()
        return IPCResponse(section=section, explanation=explanation)
    except Exception as e:
        print(f"Error: {e}")
        return IPCResponse(section=None, explanation="Unable to process the question.")


def display_response(response: IPCResponse):
    print("\n--- IPC Response ---")
    print(f"Section: {response.section if response.section else 'N/A'}")
    print(f"Explanation: {response.explanation}")


def on_button_click(b):
    user_question = text_box.value
    try:
        response = get_ipc_response(user_question)
        display_response(response)
    except Exception as e:
        print(f"Error: {e}")

text_box = widgets.Text(
    value='',
    placeholder='Ask about the Indian Penal Code',
    description='You:',
    disabled=False
)

button = widgets.Button(
    description='Ask',
    disabled=False,
    button_style='',
    tooltip='Click to ask a question about IPC',
    icon='legal'
)

button.on_click(on_button_click)


display(text_box, button)

 












 #program 10
!pip install langchain cohere wikipedia-api pydantic ipywidgets langchain-community

from langchain import PromptTemplate
# Import Cohere from langchain_community.llms
from langchain_community.llms import Cohere
from pydantic import BaseModel
from typing import Optional
import wikipediaapi
from IPython.display import display, clear_output # Import clear_output here
import ipywidgets as widgets
import getpass
import re # Import re for the regex used later

COHERE_API_KEY = getpass.getpass('Enter your Cohere API Key: ')
cohere_llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command")


def fetch_ipc_summary() -> str:
    user_agent = "IPCChatbot/1.0 (contact: myemail@example.com)"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')
    page = wiki_wiki.page("Indian Penal Code")
    if not page.exists():
        raise ValueError("The Indian Penal Code page does not exist on Wikipedia.")
    return page.text[:5000]


try:
    ipc_content = fetch_ipc_summary()
except Exception as e:
    ipc_content = "Could not fetch IPC content from Wikipedia. Please check your connection."
    print(f"Error: {e}")


class IPCResponse(BaseModel):
    section: Optional[str]
    explanation: Optional[str]


prompt_template = PromptTemplate(
    input_variables=["ipc_content", "question"],
    template="""
    You are a legal assistant chatbot specialized in the Indian Penal Code (IPC).
    Refer to the following IPC document content to answer the user's query:
    {ipc_content}
    User Question: {question}
    Provide a detailed answer, mentioning the relevant section if applicable.
    """
)


def get_ipc_response(question: str) -> IPCResponse:
    try:
        formatted_prompt = prompt_template.format(ipc_content=ipc_content, question=question)
        response = cohere_llm.predict(formatted_prompt)

        # The splitting logic here might need adjustment depending on the actual output format of the LLM
        # This is a basic attempt to extract section and explanation
        if "Section" in response:
            try:
                # Attempt to find the section number after "Section" and then the explanation after ":"
                section_match = re.search(r"Section\s*(\d+[A-Z]?)", response)
                if section_match:
                    section = section_match.group(1)
                    # Find the text after the first colon or after the section if no colon follows it immediately
                    explanation_start = response.find(section) + len(section)
                    explanation = response[explanation_start:].split(':', 1)[-1].strip()
                else:
                    # Fallback if "Section" is present but not in the expected format
                    section = None
                    explanation = response.strip()
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                section = None
                explanation = response.strip()
        else:
            section = None
            explanation = response.strip()
        return IPCResponse(section=section, explanation=explanation)
    except Exception as e:
        print(f"Error: {e}")
        return IPCResponse(section=None, explanation="Unable to process the question.")


def display_response(response: IPCResponse):
    print("\n--- IPC Response ---")
    print(f"Section: {response.section if response.section else 'N/A'}")
    print(f"Explanation: {response.explanation}")


def on_button_click(b):
    user_question = text_box.value
    # Clear previous output
    with output_area:
        clear_output(wait=True)
        try:
            response = get_ipc_response(user_question)
            display_response(response)
        except Exception as e:
            print(f"Error: {e}")

text_box = widgets.Text(
    value='',
    placeholder='Ask about the Indian Penal Code',
    description='You:',
    disabled=False,
    layout=widgets.Layout(width='80%') # Adjust width for better display
)

button = widgets.Button(
    description='Ask',
    disabled=False,
    button_style='',
    tooltip='Click to ask a question about IPC',
    icon='legal'
)

# Create an output widget to display results
output_area = widgets.Output()

button.on_click(on_button_click)

# Display the widgets and the output area
display(text_box, button, output_area)
