import PyPDF2
import spacy
import nltk
import re
import fitz  # PyMuPDF
from collections import OrderedDict

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from transformers import T5ForConditionalGeneration, T5Tokenizer

def extract_text_from_pdf2(pdf_file_path):
    text = ""
    # Open the PDF file
    with fitz.open(pdf_file_path) as pdf_document:
        # Iterate through each page
        for page_number in range(len(pdf_document)):
            # Get the page
            page = pdf_document.load_page(page_number)
            # Extract text from the page
            page_text = page.get_text()
            # Append the extracted text to the result
            text += page_text + "\n"  # Add a newline between pages if needed

    return text
def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    phone_regex = r'\b(?:\d[ -.]*){9,}\b'
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Compile the regular expressions
    phone_pattern = re.compile(phone_regex)
    email_pattern = re.compile(email_regex)
    url_pattern = re.compile(url_regex)

    doc = nlp(text)
    cleaned_text = []
    for token in doc:
        if not phone_pattern.match(token.text) and not email_pattern.match(token.text) and not url_pattern.match(token.text) and  not  token.ent_type_ in [ "GPE"]:
            cleaned_text.append(token.text)
        else:
            print(token)
    return " ".join(cleaned_text)

def generate_abstractive_summary(text):
    # Load the pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Tokenize the input text
    text = preprocess_text(text)
    inputs = tokenizer.encode("Resume:  " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary tokens and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary_sentences = summary.split(". ")
    unique_sentences = list(OrderedDict.fromkeys(summary_sentences))  # Remove duplicates while preserving order
    filtered_summary = ". ".join(unique_sentences)
    return filtered_summary

def remove_duplicates(text):
    # Split the text into sentences
    sentences = text.split(". ")

    # Remove duplicates while preserving the order
    unique_sentences = list(OrderedDict.fromkeys(sentences))

    # Join the unique sentences into a single string
    unique_text = ". ".join(unique_sentences)

    return unique_text
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        text = text.replace('\n', '')
    return text

def extract_profile_skills_projects_experience(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    profile = ""
    skills = []
    projects = []
    experience = ""

    # Increase the word limit for doc.ents
    nlp.max_length = len(text) * 2

    for sentence in doc.sents:
        if 'skills' in sentence.text.lower():
            profile = sentence.text
        elif 'projects' in sentence.text.lower():
            projects.append(sentence.text)
        elif 'work experience' in sentence.text.lower():
            experience = sentence.text

    # Extracting skills from the profile section
    for token in nlp(profile):
        if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and token.text.lower() not in ['skills']:
            skills.append(token.text)

    return profile, list(set(skills)), projects, experience

def parse_resume(pdf_file_path):
    # resume_text = extract_text_from_pdf(pdf_file_path)
    # profile, extracted_skills, extracted_projects, experience = extract_profile_skills_projects_experience(resume_text)
    #
    # print('Profile:', profile.strip())
    # print('Extracted Skills:', extracted_skills)
    # print('Extracted Projects:')
    # for project in extracted_projects:
    #     print('-', project.strip())
    # print('Work Experience:')
    # print(experience.strip())
    # return profile.strip()
    resume = extract_text_from_pdf2(pdf_file_path)
    summarized_resume = generate_abstractive_summary(resume)
    return summarized_resume

# if __name__ == '__main__':
#     pdf_path = 'candidate_140.pdf'
#     parse_resume(pdf_path)
# text = """
# Resume: assisting other partners in high value complex cases. Also in charge of preparing and drafting legal documents, such as wills, deeds, patent applications, mortgages, leases, and contracts. Duties : \uf0b7 Drafting & negotiating all legal documents for a range of UK & international clients . \uf0b7 Advising clients on commercial contracts and agreements, company law and corporate compliance. \uf0b7 Attending and representing clients at Magistrates and County Courts. \uf0b7 Planning and organising workloads in order to meet business priorities. \uf0b7 Managing the commercial and intellectual property aspects in due diligence. \uf0b7 Settling disputes and supervising any agreements . \uf0b7 Interpreting laws, rul ings and regulations. \uf0b7 Drafting, reviewing and negotiating contracts with third party suppliers . \uf0b7 Educating and advising internal departments around legal requirements . \uf0b7 Presenting and summarizing cases to judges and juries. Solicitors Office - Coventry LAWYER April 2010 \u2013 June 2010 KEY SKILLS AND COMPETENCIES \uf0b7 Proven ability to solve problems in a methodical and practical way. \uf0b7 Ability to communicate persuasively and clearly, both orally and in writing. \uf0b7 Self motivated and a proven ability to work well as part of a team. \uf0b7 Highly skilled and negotiating and debating. \uf0b7 Effective management of external advisors on any project. \uf0b7 Studying statutes, decisions, regulations, and ordinances of quasi -judicial bodies to determine the ramifications for cases . \uf0b7 Evaluating findings and developing strategies and arguments in preparation for presentation of cases. ACADEMIC QUALIFICATIONS Sparkbrook University 200 8 - 2010 BA (Hons) Law Coventry Central College 200 5 - 2008 A levels: Maths (A) English (B) Technology (B) Science (C) REFERENCES \u2013 Available on request. Copyright information - Please read \u00a9 This Lawyer resume template is the copyright of Dayjob Ltd 2012. Jobseekers may download and use this example for their own personal use to help them create their own unique lawyer resume. You are most welcome to link to any page on our site www.d ayjob.com . However this sample must not be distributed or made available on other websites without our prior permission. For any questions relating to the use of this resume template please email: info@dayjob.com . " }"""
#
# # Generate the abstractive summary
# summary = generate_abstractive_summary(text)
# print("Abstractive Summary:")
# print(summary)
def generate_summary_2(input_text):
    # Load the Pegasus model and tokenizer
    model_name = "google/pegasus-large"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Define the input text

    # Tokenize the input text
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=150, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Print the generated summary
    print("Generated Summary:")
    return (summary)
