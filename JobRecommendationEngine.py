from DataProcessor import JobDataset
from Clustering import perform_clustering, predict_cluster
from SimilarityMatcher import SimilarityMatcher

class JobRecommendationEngine:
    def __init__(self, filename, sample_size, required_cols):
        self.job_data = JobDataset(filename, sample_size, required_cols)
        self.cluster = None
        self.similarity_matcher = None


    def preprocess_and_cluster_data(self):
        processed_dataset = self.job_data.get_processed_dataset()
        job_embeddings = processed_dataset['embeddings'].to_list()
        job_embeddings = [emb.tolist() for emb in job_embeddings]  # Convert numpy arrays to lists
        self.cluster = perform_clustering(job_embeddings)

    def train_similarity_matcher(self):
        self.similarity_matcher = SimilarityMatcher(self.job_data)

    def get_similar_jobs(self, candidate_profile):
        candidate_profile = self.job_data.preprocess_text(candidate_profile)
        candidate_embedding = self.job_data.nlp(candidate_profile).vector

        candidate_cluster = predict_cluster(self.cluster, candidate_embedding)
        cluster_job_data = self.job_data.dataset[self.cluster.labels_ == candidate_cluster]

        top_matches = self.similarity_matcher.find_top_matches(candidate_embedding, cluster_job_data)
        return top_matches

# Usage example:
job_recommendation_engine = JobRecommendationEngine('Data/data job posts.csv', 200, ['RequiredQual'])
job_recommendation_engine.preprocess_and_cluster_data()
job_recommendation_engine.train_similarity_matcher()

candidate_profile = """
Well-versed in Data Structures, Algorithms & Object Oriented Programming Skills. As a technology enthusiast exploring diﬀerent technologies such as Machine Learning, Blockchain, Android Development, etc. More than three national-level scholarships holder. Innovative, creative, hardworking, and willing to contribute to the betterment of society. karunakadam2003@gmail.com 7066377652 Pune, India linkedin.com/in/karuna-kadam-94997620b github.com/karunakadam2003 EDUCATION B.Tech. MKSSS’ Cummins College of Engineering for Women 01/2021 - Present , Pune Current CGPA: 9.5 12th Grade NMV Junior College 06/2019 - 03/2020 , Pune Marks: 91.69% 10th Grade Aranyeshwar Madhyamik Vidyalaya 06/2017 - 03/2018 , Pune Marks: 93.60% WORK EXPERIENCE Contributor and Content Writer Geeks For Geeks 04/2022 - Present , Project Intern Teknogeeks 01/2022 - 03/2022 , During the internship, I got an opportunity to work on real-life problems also I got hands-on experience with c and C++ programming languages Volunteer NXT Wave CCBP 4.0 SKILLS SQL Core Java Python C, C++ PERSONAL PROJECTS Budget Tracker using Python, Tkinter Implemented a GUI based Budget Tracker project using Python and Tkinter. Spam Mail Detection using python and ML Learnt to implement machine learning techniques to the email spam ﬁltering process of the leading internet service providers Holiday Planner using Data Structures Learnt optimal use of data structures and and their application by working on real life holiday planner system ORGANIZATIONS Schneider Electric, India (03/2022 - Present) Scholar Katalyst (06/2021 - Present) Mentee Joining Hands, Delhi Non-Government Organization (01/2021 - Present) Scholar Pratham (06/2017 - Present) Volunteer, Mentee, Scholar CERTIFICATES Data Analytics Using Python IIT Kharagpur Python Programming IIT Kharagpur Deep Learning Using Python Worked on Implementation & Research Areas during online workshop conducted by ACM-W Computer Engineering Science Stream Courses
"""
top_matches = job_recommendation_engine.get_similar_jobs(candidate_profile)
for desc, title, sim in top_matches:
    print(f"Job Title: {title}")
    print(f"Job Description: {desc}")
    print(f"Cosine Similarity: {sim}")
    print()
