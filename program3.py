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
