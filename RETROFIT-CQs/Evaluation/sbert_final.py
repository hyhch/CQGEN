!pip install sentence_transformers torch
!pip install --upgrade tensorflow transformers
!pip install tensorflow-cpu
!pip install pandas openpyxl

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from google.colab import drive
from IPython.display import Audio
from numpy import sin, pi, arange
from tqdm import tqdm, trange
from openpyxl import load_workbook, Workbook
import warnings

# Suppress the UserWarning related to tensorflow and torch-xla
warnings.filterwarnings("ignore", category=UserWarning, message="tensorflow")

# Mount Google Drive
drive.mount("/content/gdrive")

def beep(frequency=440, duration=1, sampling_rate=44100):
    t = arange(sampling_rate * duration)
    waveform = sin(2 * pi * frequency * t / sampling_rate)
    return Audio(waveform, rate=sampling_rate, autoplay=True)

def main(input_file, output_file):
    df = pd.read_excel(input_file, header=None, names=['Sentence1', 'Sentence2'])  # The column 'Sentence1' contains the benchmark CQs, and the column 'Sentence2' contains the generated CQs
    print("Data read from Excel:", df.head())  # Debugging line

    num_rows_first_column = df['Sentence1'].count()
    print("Number of rows in the first column:", num_rows_first_column)  # Debugging line

    sentence1_list = df['Sentence1'].astype(str).tolist()
    sentence2_list = df['Sentence2'].astype(str).tolist()
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    print("Model loaded:", model)  # Debugging line

    output_rows = []

    for i, sentence1 in enumerate(sentence1_list):
        if i >= num_rows_first_column:
            break

        embedding1 = model.encode(sentence1, convert_to_tensor=True)
        matched_sentences = []

        for sentence2 in sentence2_list:
            embedding2 = model.encode(sentence2, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
            print(f"Similarity between '{sentence1}' and '{sentence2}': {similarity_score}")  # Debugging line

            if similarity_score >= 0.7:# set your similarity threshold
                matched_sentences.append(sentence2)

        decision = ', '.join(matched_sentences) if matched_sentences else "No match"
        output_rows.append([sentence1, decision])

    output_df = pd.DataFrame(output_rows, columns=['Sentence1', 'Decision'])
    print("Output DataFrame:", output_df.head())  # Debugging line

    output_df.to_excel(output_file, index=False)
    print("Output file generated successfully!")

# Call the main function
main('the path for your input file,
     'the path for your output file')
beep()
