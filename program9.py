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
