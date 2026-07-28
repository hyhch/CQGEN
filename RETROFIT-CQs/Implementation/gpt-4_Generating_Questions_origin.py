!pip install openai
!pip install pandas openpyxl
from IPython.display import Audio # Module to play audio in Jupyter notebooks
from numpy import sin, pi, arange
import csv
import openai # OpenAI API for GPT-based language generation
import pandas as pd
from openpyxl import load_workbook
from google.colab import drive
# Optional: Function to generate a sine wave sound at a specified frequency and duration
def beep(frequency=440, duration=1, sampling_rate=44100):
    t = arange(sampling_rate * duration)
    waveform = sin(2 * pi * frequency * t / sampling_rate)
    return Audio(waveform, rate=sampling_rate, autoplay=True)

# Function to mount Google Drive in Google Colab
def mount_drive():
    try:
        drive.mount("--specify your triples file path--")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return False
    return True

openai.api_key = '--add your api_key--'

# # Function to generate ontology competency questions using GPT-4 based on CSV row triples and a given prompt
def generate_questions(rows):
    questions = []
    for row in rows:
        #complete_prompt = f"Subject: {row[0]}, Predicate: {row[1]}, Object: {row[2]}" # Create prompt from CSV row data
        complete_prompt = f"{', '.join(row)}?"
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4', # Specify GPT-4 model
                messages=[
                    {"role": "system", "content": "As an ontology engineer, Provide competency questions focused on the context provided; avoid using narrative questions. competency questions are the questions that outline the scope of an ontology and provide an idea about the knowledge that needs to be entailed in the ontology."},
                    {"role": "user", "content": complete_prompt}
                ],
                max_tokens=4166,
                n=1,
                stop=None,
                temperature=1,
            )
            question = response['choices'][0]['message']['content'].strip() # Extract response content
            questions.append(question)
        except Exception as e:
            print(f"Error generating question for row {row}: {e}")
            questions.append("Error generating question")
    return questions

# Function to read CSV file, generate questions, and save to Excel file
def generate_questions_from_csv(file_path, output_file):
    try:
        df = pd.read_csv(file_path, sep='\t')  # Read CSV file with tab delimiter
    except Exception as e:
        print(f"Error reading CSV file: {e}")  # Handle file reading errors
        return

    df_chunks = [df[i:i+10] for i in range(0, df.shape[0], 5)]  # Split data into chunks for processing

    for i, chunk in enumerate(df_chunks):
        rows = chunk.values.tolist()  # Convert chunk to list format for question generation
        questions = generate_questions(rows)  # Generate questions for each row in the chunk
        chunk['Question'] = questions  # Append questions as a new column in the chunk

        try:
            book = load_workbook(output_file)  # Load existing Excel workbook
            writer = pd.ExcelWriter(output_file, engine='openpyxl')
            writer.book = book  # Assign workbook to writer
            chunk.to_excel(writer, sheet_name=f'Sheet{i+1}', index=False)  # Write chunk to Excel sheet
            writer.save()  # Save changes to the workbook
            writer.close()  # Close writer to release file
        except Exception as e:
            print(f"Error writing to Excel file: {e}")  # Handle Excel writing errors

# Main process: mount Google Drive, generate questions from CSV, and save results
if mount_drive():
    
    file_path = 'input_file' # Define the triple file path
    output_questionsFile = 'output_file'#  Define the generated CQs file path
    
       generate_questions_from_csv(file_path, output_questionsFile)
    
# Play a beep sound after completion
beep()

