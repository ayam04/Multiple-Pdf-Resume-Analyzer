# Multiple-Pdf-Resume-Analyzer

This code is designed to rank resumes based on their similarity to a given keyword using TensorFlow and Universal Sentence Encoder. It provides a way to analyze multiple PDF resumes and determine which ones match the keyword the most.

## Prerequisites

Before running this code, make sure you have the following:

- [ ] TensorFlow and TensorFlow Hub installed
- PyPDF2 library installed
- PDF resumes stored in a folder

## How to Use

1. Set the folder path containing the PDF resumes in the `folder_path` variable.
2. If the vector embedding data already exists, the code will load it from `vector_data.pickle`. Otherwise, it will extract text from each resume, convert it into vector embeddings using Universal Sentence Encoder, and save the embeddings in `vector_data.pickle` for future use.
3. Enter the keyword you want to search for when prompted.
4. The code will calculate the similarity scores between the keyword and each resume using cosine similarity. The higher the score, the more similar the resume is to the keyword.
5. The ranked resumes will be displayed, showing the file name, rank, and similarity score.

## Conclusion

This code allows you to efficiently analyze and rank resumes based on their similarity to a given keyword. It can be a valuable tool for quickly identifying the most relevant resumes for a specific job or search criteria. Feel free to modify and adapt the code to suit your specific requirements and integrate it into your workflow.
