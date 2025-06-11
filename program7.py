!pip install transformers 
# Import the summarization pipeline 
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
