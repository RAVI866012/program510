!pip install langchain cohere langchain-community --quiet 
# 🚀 Import necessary libraries 
import cohere 
import getpass 
from langchain import PromptTemplate 
from langchain.llms import Cohere 
# 🚀 Load the text file 
file_path = "Teaching.txt"  # 🔁 Change this to the path of your file 
try: 
  with open(file_path, "r", encoding="utf-8") as file: 
  text_content = file.read() 
  print("✅ File loaded successfully!") 
except Exception as e: 
  print("❌ Error loading file:", str(e)) 
  text_content = ""  # Avoid crashing later 
# 🚀 Set up Cohere API key 
COHERE_API_KEY = getpass.getpass("🔑 Enter your Cohere API Key: ") 
# 🚀 Initialize the Cohere model using LangChain 
cohere_llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command") 
# 🚀 Define prompt template 
template = """ 
          You are an AI assistant helping to summarize and analyze a text document. 
          Here is the document content: 
          {text} 
          �
          � Summary: Provide a concise summary of the document. 
          �
          � Key Takeaways: List 3 important points from the text. 
          �
          � Sentiment Analysis: Determine if the sentiment of the document is Positive, Negative, or Neutral. 
          """ 
prompt_template = PromptTemplate(input_variables=["text"], template=template) 
# 🚀 Generate response 
if text_content.strip(): 
  formatted_prompt = prompt_template.format(text=text_content) 
  response = cohere_llm.predict(formatted_prompt) 
    # 🚀 Display the result 
  print("\n📌 **Formatted Output** 📌\n") 
  print(response) 
else: 
  print("⚠️ No text to analyze.")
