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
