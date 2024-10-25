import streamlit as st
import pickle
import re
import nltk

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))


def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


# Web app
def main():
    st.set_page_config(page_title="Resume Screening App", layout="wide")

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Choose your action", ["Home", "Upload Resume", "About"])

    if app_mode == "Home":
        st.title("Welcome to the Resume Screening App")
        # st.image("resume_sample.png", caption="Sample Resume Format", use_column_width=True)
        st.write("This app helps to screen resumes based on specific job categories.")
        st.write("### How to Use:")
        st.write("1. Navigate to the 'Upload Resume' section from the sidebar.")
        st.write("2. Upload your resume in either .txt or .pdf format.")
        st.write("3. View the predicted job category based on the content of your resume.")
        st.write("4. Explore the 'About' section to learn more about this application.")

    elif app_mode == "Upload Resume":
        st.title("Resume Upload")
        uploaded_file = st.file_uploader('Upload Resume (TXT or PDF)', type=['txt', 'pdf'])

        if uploaded_file is not None:
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidfd.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            st.success("Resume successfully processed!")
            st.write("### Predicted Category:")
            st.write(f"**{category_name}**")
            st.write("You can explore more job categories on the home page.")

    elif app_mode == "About":
        st.title("About This App")
        st.write("This application is designed to assist employers in screening resumes based on job roles.")
        st.write("It uses machine learning models to analyze the content of resumes and categorize them accordingly.")
        st.write("### Key Features:")
        st.write("- User-friendly interface.")
        st.write("- Supports both text and PDF formats.")
        st.write("- Provides immediate feedback on the predicted job category.")


# Run the app
if __name__ == "__main__":
    main()
