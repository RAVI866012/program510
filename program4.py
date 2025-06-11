import nltk 
nltk.download('punkt') 
import string 
from nltk.tokenize import word_tokenize 
from transformers import pipeline 
import gensim.downloader as api 
# Download necessary NLTK data 

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
            return generator(prompt, max_length=100, num_return_sequences=1, truncation=True)[0]['generated_text'] 
        except Exception as e: 
            return f"Error generating response: {e}" 
    return "No response generated (model loading failed)." 
# Example usage 
original_prompt = "Who is king." 
key_term = "king" 

enriched_prompt = replace_keyword(original_prompt, key_term, word_vectors) if word_vectors else original_prompt 
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
print(f"Enriched Prompt Response Length: {len(enriched_response)} ")


