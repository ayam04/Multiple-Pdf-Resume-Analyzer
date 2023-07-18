import tensorflow as tf
import tensorflow_hub as hub
import PyPDF2
import os
import pickle
import time

model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(model_url)

# Dictionary to store vectorized embeddings
resume_embedding_dict = {}

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
def calculate_similarity(keyword_embedding, resume_embedding):
    similarity_score = tf.reduce_sum(tf.multiply(keyword_embedding, resume_embedding))
    return similarity_score

# Accept the folder path containing resumes
folder_path = r"C:\\Users\\ayamu\\python programs\\pdf_keyword_searcher\\pdfs"

# Load or create the vector embedding data
vector_data_path = "vector_data.pickle"
if os.path.exists(vector_data_path):
    with open(vector_data_path, "rb") as file:
        resume_embedding_dict = pickle.load(file)
else:
    for filename in os.listdir(folder_path):
        resume_file = os.path.join(folder_path, filename)
        resume_text = extract_text_from_pdf(resume_file)
        resume_embedding = embed([resume_text])[0]
        resume_embedding_dict[resume_file] = resume_embedding
    # Save the vector embedding data
    with open(vector_data_path, "wb") as file:
        pickle.dump(resume_embedding_dict, file)

keyword = input("Enter a keyword to search for: ")
keyword_embedding = embed([keyword])[0]  # Convert keyword to a list
# Save the keyword embedding
keyword_embedding_path = "keyword_embedding.pickle"
with open(keyword_embedding_path, "wb") as file:
    pickle.dump(keyword_embedding, file)

# Process resumes and calculate similarity scores
resumes = []
for resume_file, resume_embedding in resume_embedding_dict.items():
    similarity_score = calculate_similarity([keyword_embedding], [resume_embedding])  # Convert keyword and resume_embedding to lists
    resumes.append((resume_file, similarity_score))
# Sort the resumes based on similarity score
resumes.sort(key=lambda x: x[1], reverse=True)
start = time.time()

# Display the ranked resumes
print("Ranked Resumes:")
for i, (resume_file, similarity_score) in enumerate(resumes):
    print(f"Rank {i+1}: {resume_file} (Similarity Score: {similarity_score:.2f})")
end = time.time()
print(end - start)
