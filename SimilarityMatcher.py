from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatcher:
    def __init__(self, job_data):
        self.job_data = job_data

    def calculate_cosine_similarity(self, candidate_embedding, job_embedding):
        return cosine_similarity([candidate_embedding], [job_embedding])[0][0]

    def find_top_matches(self, candidate_embedding, cluster_job_data):
        similarities = []
        for idx, row in cluster_job_data.iterrows():
            # job_embedding = self.job_data.nlp(row['jd']).vector
            similarity = self.calculate_cosine_similarity(candidate_embedding, row['embeddings'])
            similarity_percentage = "{:.2f}%".format(similarity * 100)
            similarities.append([row['Title'], row['Company'], similarity_percentage])
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:10]

# Usage example:
# # Initialize SimilarityMatcher with job data
# similarity_matcher = SimilarityMatcher(job_data)
#
# # Compute cosine similarity between candidate profile and job descriptions in the cluster
# top_matches = similarity_matcher.find_top_matches(candidate_embedding, cluster_job_data)
#
# # Print top 5 job descriptions with highest similarity
# for desc, title, sim in top_matches:
#     print(f"Job Title: {title}")
#     print(f"Job Description: {desc}")
#     print(f"Cosine Similarity: {sim}")
#     print()
