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
        num_none_values = sum(1 for emb in job_embeddings if emb is None)
        print("Number of None values in job_embeddings:", num_none_values)
        job_embeddings = [emb.tolist() for emb in job_embeddings]  # Convert numpy arrays to lists
        self.cluster = perform_clustering(job_embeddings)
        #self.cluster =  perform_clustering(self.job_data.tfidf_matrix)

    def train_similarity_matcher(self):
        self.similarity_matcher = SimilarityMatcher(self.job_data)

    def get_similar_jobs(self, candidate_profile):
        candidate_profile = self.job_data.preprocess_text(candidate_profile)
        candidate_embedding = self.job_data.nlp(candidate_profile).vector
        #candidate_embedding = self.job_data.tfidf_vectorizer.transform([candidate_profile])
        candidate_cluster = predict_cluster(self.cluster,candidate_embedding)

        cluster_job_data = self.job_data.dataset[self.cluster.labels_ == candidate_cluster]

        top_matches = self.similarity_matcher.find_top_matches(candidate_embedding, cluster_job_data)
        return top_matches

# Usage example:
# job_recommendation_engine = JobRecommendationEngine('Data/data job posts.csv', 2000, ['Title','RequiredQual'])
# job_recommendation_engine.preprocess_and_cluster_data()
# job_recommendation_engine.train_similarity_matcher()

# candidate_profile = """
# I am a seasoned lawyer with over 10 years of experience specializing in corporate law. My expertise lies in mergers and acquisitions, contract negotiation, and intellectual property law. I have successfully represented multinational corporations in high-stakes litigation cases and have a proven track record of delivering favorable outcomes for my clients.
#
# Throughout my career, I have demonstrated strong analytical skills, attention to detail, and the ability to navigate complex legal issues. I am highly proficient in drafting legal documents, conducting legal research, and providing strategic legal advice to clients.
#
# In addition to my legal expertise, I possess excellent communication and negotiation skills, which enable me to effectively advocate for my clients' interests. I am committed to upholding the highest ethical standards and am dedicated to achieving the best possible results for my clients.
#
# I hold a Juris Doctor (JD) degree from [Law School Name] and I am licensed to practice law in [State/Country]. I am passionate about the law and am continuously seeking opportunities for professional growth and development in the legal field.
# """
# top_matches = job_recommendation_engine.get_similar_jobs(candidate_profile)
# for desc, title,company, sim in top_matches:
#
#     print(f"Job Title: {title}")
#     print(f"Company: {company}")
#     #print(f"Job Description: {desc}")
#     print(f"Cosine Similarity: {sim}\n")
