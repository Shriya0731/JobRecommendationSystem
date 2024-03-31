#Step – 1(import necessary library)
from flask import (Flask, render_template, request, redirect, session, jsonify)
import DBConnection as db
import backend
import send_mail
from JobRecommendationEngine import JobRecommendationEngine
from ResumeParser import parse_resume
global top_matches
#Step – 2 (configuring your application)
app = Flask(__name__)
app.secret_key = 'shriya'

#step – 3 (creating a dictionary to store information about users)
user = {"username": "abc", "password": "xyz"}
job_recommendation_engine = JobRecommendationEngine('Data/data job posts.csv', 2000, ['Title','RequiredQual'])
job_recommendation_engine.preprocess_and_cluster_data()
job_recommendation_engine.train_similarity_matcher()

@app.route('/', methods=['POST', 'GET'])
def start():
    return render_template("index.html")

@app.route('/addresume')
def addresume():
    return render_template("add_resume.html")

@app.route('/addjd')
def addjd():
    return render_template("add_jd.html")

@app.route('/show_jobs')
def show_jobs():
    global top_matches
    return render_template("jobs.html", jobs = top_matches)

@app.route('/getjobs', methods=['POST'])
def getjobs():
    global top_matches
    print("hello")
    if(request.method == 'POST'):
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

    # You can save the file to a desired location
    file.save('Data/' + file.filename)
    extracted_skills = parse_resume('Data/' + file.filename)

    candidate_profile = extracted_skills
    print(candidate_profile)
    top_matches = job_recommendation_engine.get_similar_jobs(candidate_profile)
    print(top_matches)
    for title,company, sim in top_matches:

        print(f"Job Title: {title}")
        print(f"Company: {company}")
        #print(f"Job Description: {desc}")
        print(f"Cosine Similarity: {sim}\n")

    return redirect('/show_jobs')


@app.route('/upload_resume', methods=['POST'])
def getmatchingjobs():
    if (request.method == 'POST'):
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        # You can save the file to a desired location
        file.save('Data/' + file.filename)
        extracted_skills = parse_resume('Data/' + file.filename)

        # Return the matching jobs as JSON response
        return jsonify({'extracted_skills': extracted_skills})
    #return render_template("index.html")
    #return redirect('/addresume')
    #return render_template("job-desc.html",usr=db.getrole())

@app.route('/add', methods=['POST'])
def addjdtodb():
    if (request.method == 'POST'):
        role = request.form.get('role')
        desc = request.form.get('desc')
        db.insert_new_jd(role,desc)
    #return render_template("index.html")
    return redirect('/displayrole')
    #return render_template("job-desc.html",usr=db.getrole())

@app.route('/display')
def display():
    return render_template("table.html",usr=db.display_table())

#
@app.route('/showprofiles', methods=['POST'])
def showprofiles():
    if (request.method == 'POST'):
        title = request.form.get('jobtitle')
        print(title)
        job_desc = db.getjobdesc(title)
        print(job_desc)
        skilldataset = backend.cleanprofilesdataset()
        backend.findsimilarityscore(job_desc,skilldataset)

    matchedprofiles = db.getmatchedprofiles()
    if (job_desc!=False):
        return render_template("profiles.html",usr=matchedprofiles,title=title,desc=job_desc[0])
    else:
        return render_template("noprofiles.html")

@app.route('/sendmails', methods = ['POST', 'GET'])
def sendmails():
    tot= int(request.form.get('tot'))
    title = request.form.get('title')
    print("For position "+title)
    print("Total mails to be sent:",tot)
    data = db.getmails_name()
    print("Mail Sent to:")
    print(data[1])
    send_mail.multiple_mails(data[0], data[1],tot,title)
    return render_template("mailsent.html")

@app.route('/displayrole')
def displayroles():
    #return db.getrole()
    # return render_template("table.html",usr=db.display_table())
    return render_template("job-desc.html",usr=db.getrole())


if __name__ == '__main__':
    app.run(debug=True)
