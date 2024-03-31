from sklearn.cluster import KMeans
def perform_clustering(job_embeddings):
    num_clusters = 3  # You can choose the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42,n_init=10)
    kmeans.fit(job_embeddings)
    return kmeans

def predict_cluster(cluster,candidate_embedding):
    candidate_cluster = cluster.predict([candidate_embedding])[0]
    return candidate_cluster

# # Filter job descriptions belonging to the same cluster
#
#
#
# job_data = JobDataset('data job posts.csv', 200,['RequiredQual'])
# dataset = job_data.get_processed_dataset()
#
# # Compute the embedding for the candidate profile
# candidate_profile = """
# Well-versed in Data Structures, Algorithms & Object Oriented Programming Skills. As a technology enthusiast exploring diﬀerent technologies such as Machine Learning, Blockchain, Android Development, etc. More than three national-level scholarships holder. Innovative, creative, hardworking, and willing to contribute to the betterment of society. karunakadam2003@gmail.com 7066377652 Pune, India linkedin.com/in/karuna-kadam-94997620b github.com/karunakadam2003 EDUCATION B.Tech. MKSSS’ Cummins College of Engineering for Women 01/2021 - Present , Pune Current CGPA: 9.5 12th Grade NMV Junior College 06/2019 - 03/2020 , Pune Marks: 91.69% 10th Grade Aranyeshwar Madhyamik Vidyalaya 06/2017 - 03/2018 , Pune Marks: 93.60% WORK EXPERIENCE Contributor and Content Writer Geeks For Geeks 04/2022 - Present , Project Intern Teknogeeks 01/2022 - 03/2022 , During the internship, I got an opportunity to work on real-life problems also I got hands-on experience with c and C++ programming languages Volunteer NXT Wave CCBP 4.0 SKILLS SQL Core Java Python C, C++ PERSONAL PROJECTS Budget Tracker using Python, Tkinter Implemented a GUI based Budget Tracker project using Python and Tkinter. Spam Mail Detection using python and ML Learnt to implement machine learning techniques to the email spam ﬁltering process of the leading internet service providers Holiday Planner using Data Structures Learnt optimal use of data structures and and their application by working on real life holiday planner system ORGANIZATIONS Schneider Electric, India (03/2022 - Present) Scholar Katalyst (06/2021 - Present) Mentee Joining Hands, Delhi Non-Government Organization (01/2021 - Present) Scholar Pratham (06/2017 - Present) Volunteer, Mentee, Scholar CERTIFICATES Data Analytics Using Python IIT Kharagpur Python Programming IIT Kharagpur Deep Learning Using Python Worked on Implementation & Research Areas during online workshop conducted by ACM-W Computer Engineering Science Stream Courses
# """
# candidate_profile = job_data.preprocess_text(candidate_profile)
# candidate_embedding = job_data.nlp(candidate_profile).vector
#
# # Convert embeddings column to 2D array
# job_embeddings = dataset['embeddings'].to_list()
# job_embeddings = [emb.tolist() for emb in job_embeddings]  # Convert numpy arrays to lists
#
# clusters = perform_clustering(job_embeddings)
# candidate_cluster = predict_cluster(clusters,candidate_embedding)
# cluster_job_data = dataset[clusters.labels_ == candidate_cluster]
#
# # Compute cosine similarity between candidate profile and job descriptions in the cluster
# similarities = []
# for idx, row in cluster_job_data.iterrows():
#     job_embedding = job_data.nlp(row['jd']).vector
#     similarity = cosine_similarity([candidate_embedding], [job_embedding])[0][0]
#     similarities.append((row['jd'], row['Title'], similarity))
#
# # Print top 5 job descriptions with highest similarity
# top_matches = sorted(similarities, key=lambda x: x[2], reverse=True)[:5]
# for desc, title, sim in top_matches:
#     print(f"Job Title: {title}")
#     print(f"Job Description: {desc}")
#     print(f"Cosine Similarity: {sim}")
#     print()
