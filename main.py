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

 