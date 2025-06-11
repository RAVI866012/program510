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
        f"People often associate {seed_word} with {similar_words[2][0]} and {similar_words[3][0]}.", 
        f"A story about {seed_word} would be incomplete without {similar_words[1][0]} and {similar_words[3][0]}.", 
        ] 
    paragraph = " ".join(random.sample(paragraph_templates, len(paragraph_templates))) 
    return paragraph 
# Example usage 
seed_word = input("Enter a seed word: ") 
print("\nGenerated Paragraph:\n") 
print(generate_paragraph(seed_word))
