import PyPDF2
import spacy

def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
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
    return extract_text_from_pdf(pdf_file_path)

# if __name__ == '__main__':
#     pdf_path = 'candidate_140.pdf'
#     parse_resume(pdf_path)
