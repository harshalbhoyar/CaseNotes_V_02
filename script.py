import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import io
from PIL import Image
from datetime import date
from transformers import pipeline
import numpy as np

global csv_text
api_key = "HERE_OPENAI_KEY"

def process(data, query):
    text_splitter = CharacterTextSplitter(chunk_size=350, chunk_overlap=0)
    texts = text_splitter.create_documents([data])
    chain = load_qa_chain(OpenAI(temperature=0, openai_api_key= api_key, max_tokens=1000), chain_type="stuff")
    answer = chain.run(input_documents=texts, question=query)



    # List of known variations of "I don't know" answers
    known_variations = [
        "i don't know.",
        "i'm sorry, i don't know.",
        "no, i don't know."
    ]


    return answer

def create_stacked_boxes_image(categories):

    # Create a BytesIO object to store the image binary data
    img_stream = io.BytesIO()

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(categories) + 1)

    # Loop through categories and create boxes with borders
    for i, category in enumerate(categories):
        rect = plt.Rectangle((1, i + 0.5), 8, 1, color='lightblue', fill=False, linewidth=2, edgecolor='white')
        ax.add_patch(rect)
        ax.text(5, i + 1, category, va='center', ha='center', color='black', fontsize=12)

    ax.set_title('Leading indicators of non-persistency or discontinuation of medications')

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot to the BytesIO object
    plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
    img_stream.seek(0)  # Reset stream position to the beginning

    # Close the plot
    plt.close()

    return img_stream


def create_percentage_bar_charts(input_string):
    categories = re.split(r'\d+\.\s+', input_string)[1:]
    category_data = []

    max_percentage = 0
    all_figures = []

    # Sort categories based on the number of subcategories
    categories.sort(key=lambda category: len(category.strip().split('\n')[1:]))

    # Select the top 2 categories with the most subcategories
    top_categories = categories[-2:]

    fig, axes = plt.subplots(len(top_categories), 1, figsize=(10, 4 * len(top_categories)))

    for category_idx, category in enumerate(top_categories):
        lines = category.strip().split('\n')
        category_name = lines[0].strip()
        subcategories = []
        percentages = []

        for line in lines[1:]:
            match = re.match(r'\s*(\w+)\)\s+([^:]+):\s+(\d+)%', line)
            if match:
                subcategories.append(match.group(2))
                percentage = int(match.group(3))
                percentages.append(percentage)
                max_percentage = max(max_percentage, percentage)

        sorted_indices = sorted(range(len(subcategories)), key=lambda k: percentages[k], reverse=True)
        sorted_subcategories = [subcategories[i] for i in sorted_indices]
        sorted_percentages = [percentages[i] for i in sorted_indices]

        ax = axes[category_idx]
        ax.barh(sorted_subcategories, sorted_percentages, color='skyblue')
        ax.set_xlabel('Percentage of Patients (%)')
        ax.set_ylabel('Subcategories')
        ax.set_title(category_name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        ax.set_yticks(range(len(sorted_subcategories)))
        ax.set_yticklabels([textwrap.fill(label, 20) for label in sorted_subcategories])
        ax.set_xlim(0, max_percentage + 10)

    plt.tight_layout()  # Adjust the layout to prevent overlap

    # Save the figure as a single image
    combined_image_path = 'combined_percentage_bar_charts.png'
    plt.savefig(combined_image_path)

    plt.close(fig)  # Close the figure after saving

    return combined_image_path


def percentage_bar_charts(input_string):
    categories = re.split(r'\s*,\s*', input_string)
    category_data = []

    max_percentage = 0
    all_figures = []

    fig, ax = plt.subplots(figsize=(7, len(categories) * 1.3))

    subcategories = []
    percentages = []

    for category in categories:
        match = re.match(r'([^:]+):\s+(\d+)%', category)
        if match:
            subcategory_name = match.group(1)
            subcategories.append(subcategory_name)
            percentage = int(match.group(2))
            percentages.append(percentage)
            max_percentage = max(max_percentage, percentage)

    sorted_indices = sorted(range(len(subcategories)), key=lambda k: percentages[k], reverse=True)
    sorted_subcategories = [subcategories[i] for i in sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]

    # Increase the width of the bars and add spacing between them
    bar_width = 0.6
    gap_between_bars = 0.2
    positions = range(len(sorted_subcategories))

    # Define a list of colors for the bars
    bar_colors = ['dodgerblue', 'green', 'orange', 'purple', 'pink', 'blue', 'red', 'gray']

    bars = ax.barh(positions, sorted_percentages, color=bar_colors, height=bar_width, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels([textwrap.fill(label, 20) for label in sorted_subcategories])
    #ax.set_xticklabels([f'{label}% ' for label in ax.get_xticks()])
    ax.set_xlabel('% of Patients')
    ax.set_ylabel('')
    ax.set_title('Top Indicators')
    #ax.set_xlim(0, max_percentage + 10)
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-gap_between_bars, len(sorted_subcategories) - 1 + gap_between_bars)
    ax.invert_yaxis()

    # Annotate bars with percentage values
    for bar, percentage in zip(bars, sorted_percentages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{percentage}%', va='center')

    plt.tight_layout()  # Adjust the layout to prevent overlap

    # Save the figure as a single image
    combined_image_path = 'combined_percentage_bar_chart.png'
    plt.savefig(combined_image_path)

    plt.close(fig)  # Close the figure after saving

    return combined_image_path, bar_colors  # Return the colors as well


def create_percentage_bar_chartss(input_string, color_list):
    categories = re.split(r'\d+\.\s+', input_string)[1:]
    category_data = []

    max_percentage = 0
    all_figures = []

    # Sort categories based on the number of subcategories
    categories.sort(key=lambda category: len(category.strip().split('\n')[1:]),
                    reverse=True)  # Reverse the sorting order

    # Select the top category with the most subcategories
    top_category = categories[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    lines = top_category.strip().split('\n')
    category_name = lines[0].strip()
    subcategories = []
    percentages = []

    for line in lines[1:]:
        match = re.match(r'\s*(\w+)\)\s+([^:]+):\s+(\d+)%', line)
        if match:
            subcategories.append(match.group(2))
            percentage = int(match.group(3))
            percentages.append(percentage)
            max_percentage = max(max_percentage, percentage)

    sorted_indices = sorted(range(len(subcategories)), key=lambda k: percentages[k], reverse=True)
    sorted_subcategories = [subcategories[i] for i in sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]

    # Increase the width of the bars and add spacing between them
    bar_width = 0.6
    gap_between_bars = 0.2
    positions = range(len(sorted_subcategories))

    bars = ax.bar(positions, sorted_percentages, color=color_list[1], width=bar_width, edgecolor='gray', linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([textwrap.fill(label, 20) for label in sorted_subcategories], rotation=45, ha='right')
    ax.set_ylabel('% of Mentions')
    ax.set_xlabel('')
    ax.set_title(category_name)
    ax.set_xlim(-gap_between_bars, len(sorted_subcategories) - 1 + gap_between_bars)
    ax.set_ylim(0, max_percentage + 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate bars with percentage values
    for bar, percentage in zip(bars, sorted_percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percentage}%', ha='center')

    plt.tight_layout()  # Adjust the layout to prevent overlap

    # Save the figure as a single image
    combined_image_path = 'combined_percentage_bar_chartss.png'
    plt.savefig(combined_image_path)

    plt.close(fig)  # Close the figure after saving

    return combined_image_path

#New Category added
def create_percentage_bar_charts(input_string, color_list):
    categories = re.split(r'\d+\.\s+', input_string)[1:]
    category_data = []

    max_percentage = 0
    all_figures = []

    # Sort categories based on the number of subcategories
    categories.sort(key=lambda category: len(category.strip().split('\n')[1:]), reverse=True)  # Reverse the sorting order

    # Select the top category with the most subcategories
    top_category = categories[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    lines = top_category.strip().split('\n')
    category_name = lines[0].strip()
    subcategories = []
    percentages = []

    for line in lines[1:]:
        match = re.match(r'\s*(\w+)\)\s+([^:]+):\s+(\d+)%', line)
        if match:
            subcategories.append(match.group(2))
            percentage = int(match.group(3))
            percentages.append(percentage)
            max_percentage = max(max_percentage, percentage)

    sorted_indices = sorted(range(len(subcategories)), key=lambda k: percentages[k], reverse=True)
    sorted_subcategories = [subcategories[i] for i in sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]

    # Increase the width of the bars and add spacing between them
    bar_width = 0.6
    gap_between_bars = 0.2
    positions = range(len(sorted_subcategories))

    bars = ax.bar(positions, sorted_percentages, color=color_list[0], width=bar_width, edgecolor='gray', linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([textwrap.fill(label, 20) for label in sorted_subcategories], rotation=45, ha='right')
    ax.set_ylabel('% of Mentions')
    ax.set_xlabel('')
    ax.set_title(category_name)
    ax.set_xlim(-gap_between_bars, len(sorted_subcategories) - 1 + gap_between_bars)
    ax.set_ylim(0, max_percentage + 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate bars with percentage values
    for bar, percentage in zip(bars, sorted_percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percentage}%', ha='center')

    plt.tight_layout()  # Adjust the layout to prevent overlap

    # Save the figure as a single image
    combined_image_path = 'combined_percentage_bar_charts.png'
    plt.savefig(combined_image_path)

    plt.close(fig)  # Close the figure after saving

    return combined_image_path


def plot_categories_in_boxes(input_string):
    categories = [category.strip("• ") for category in input_string.strip().split("\n")]
    num_categories = len(categories)

    fig, ax = plt.subplots(figsize=(6, num_categories * 0.5))
    ax.set_xlim(0, 10)  # Adjust the x-axis limits based on your preference
    ax.set_ylim(0, num_categories)
    ax.axis('off')  # Turn off axis

    for i, category in enumerate(categories):
        ax.text(0.5, num_categories - i - 0.5, category, va='center', ha='left', fontsize=12)
        ax.add_patch(plt.Rectangle((0.2, num_categories - i - 0.75), 9.2, 0.5, fill=False, color='black'))

    plt.tight_layout()

    # Save the plot as an image
    image_path = "categories_box_plot.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

    return image_path

relevent_column = "Which column has information about patient reviews/patient case notes/patient history notes etc? Your answer should only be the column name."

def extract_questions_and_answers(text):
    questions = []
    answers = []

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(' Answer: ')
            if len(parts) == 2:
                questions.append(parts[0])
                answers.append(parts[1])

    return questions, answers

def process_csv_with_columns(upload):

    # Read the uploaded CSV file into a Pandas DataFrame
    data = pd.read_csv(upload)
    data_head = data.head()
    csv_text = data_head.to_csv(index=False)

    outputs = []

    df = pd.DataFrame()

    column_name = process(csv_text, relevent_column)

    column_name = str(column_name).strip()

    df = data[[column_name]]

    #For reading file name and type

    # format_of_file_location = upload.name
    # file_name = format_of_file_location.split("/")
    # name = file_name[-1]
    # str1 = f"{name} has been successfully uploaded"


    # Add a 'patient_id' column with row numbers
    df['patient_id'] = df.index + 1

    # Move the 'patient_id' column to be the first column
    columns1 = ['patient_id'] + [col for col in df.columns if col != 'patient_id']
    df = df[columns1]

    # Randomly sample 10 rows without replacement
    #df = df.sample(n=10)

    df = df.head(10)

    print(df.head())
    # Calculate the total number of rows using the shape attribute
    total_patients = df.shape[0]

    print(total_patients)

    # # Split the DataFrame into smaller DataFrames with 10 rows each
    # chunk_size = 10
    # num_chunks = len(data) // chunk_size
    # remainder = len(data) % chunk_size

    # dataframes = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    # if remainder > 0:
    #     dataframes.append(data.iloc[num_chunks*chunk_size:])

    csv_text = df.to_csv(index=False)


    Question1 = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    #Question1 = """
    #Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    #Provide the name of each reason and the number of patients who fall into that category, in descending order.
    #"""

    #Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answer = process(csv_text, Question1) + " "

    print(answer)

    Question1_Alter = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Who are the top 2 patients who either already switched their medication or are most likely to switch or discontinue their medication?
    For Example one patient mentioned 'their doctor decided to switch the patient's medication to Brand B'
    Also mention, What might be the probable reasons for this. While refereing any patient e.g. Patient 6 add Prefix "P" means "Patient 6" would be denoted as "Patient P6". If you are refering
    only one patient like P6 in response please consider it as singular dont use words like them and thier for such cases.
    """

    answer_Alter = process(csv_text, Question1_Alter)

    QuestionX = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group with patient id.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    #Question1 = """
    #Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    #Provide the name of each reason and the number of patients who fall into that category, in descending order.
    #"""

    #Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answerX = process(csv_text, QuestionX) + " "

    print(answerX)

    Question2 = "What is the name of all category in this data? provide answer in bullet points"
    answer2 = process(answer,Question2)

    #answer2 = plot_categories_in_boxes(str(answer2))

    Question3 = "Please calculate the percentage of patients for each reason and replace the number of patients with their respective percentages.Total number of patients are : " + str(total_patients)
    answer3 = process(answer, Question3)

    if "0%" in str(answer3):
      answer3 = str(answer3).replace(", Discontinuation: 0%.", ".")

    print("answer3" + str(answer3))


    Question5 = """For each category mentioned below what all brief insights mentioned in the overall data? \n Categories : \n"""  + str(answer) + """.
    Your insights should be in subcategories format and should be unique.
    I am providing you with on sample example. Your answer should be in similiar format. Do not take data from below example, only take the format information.
    For Example : \n
    1. Cost
        a) out of pocket expense of $2k / month: Number of patients - 1
        b) Part D plan do not qualify for copay program: Number of patients - 1
        c) not covered by their insurance: Number of patients - 1
        d) very expensive: Number of patients - 3

    2. Side effects
        a) Nausea -4
        b) Bone Pain -2
        c) Abdominal bloating -1
        d) Body pain - 1
        e) Headache - 1

    \n

    Remember: You should also provide number of patient that fit into respective subcategory based on the data.
    """
    answer5 = process(csv_text, Question5)

    print('answer5 :' + str(answer5))

    Question6 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the ‘"""+column_name+"""’ column, List down the 3 important questions to understand the insights along with their answers,
    excluding information about """ + answer2 + """ .\n
    The questions should aim to uncover patterns and trends in the data that would be useful for the Data Analysis team.?
    Your answers should be in below format (Only use the format, do not copy from this):\n
    1. Your question will come here? Answer: Your answer will come here.
    """
    answer6 = process(csv_text, Question6)

    Question7 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the Sub-Categories mentioned below,
    please carefully find the number of patients for each subcategory based on the data provided.
    You should double check your answer. Your answer should be correct.
    Categories and Sub-Categories : """+answer5+ """. \n
    Do not mention patient_id in your answer just number of patients. \n
    Please do not miss any sub-categories. Just do a double check if you have included all the sub-categories. \n
    Your answers should be in below format (Only use the format, do not copy from this)):\n
    1. Cost
        a) out of pocket expense of $2k / month : Number of patients - 1
        b) Part D plan do not qualify for copay program : Number of patients - 1
        c) not covered by their insurance : Number of patients - 1
        d) very expensive : Number of patients - 1

    """
    answer7 = process(csv_text, Question7)

    print(answer7)

    Question8 = """
    You are a data analyst. You have been provided with some categories and sub-categories and their respective number of patients.
    Please calculate the percentage of patients for each Sub-Category and
    replace the number of patients with their respective percentages.Total number of patients are : """ + str(total_patients) +""".
    To caluclate the percentages you need to first divide number of patients written with Total number of patients provided to you.
    After that multiple them with 100 and add a '%' sign.
    Your final answer should be in the exact below format, I am also giving you an example. Please double check the format, the whitespaces etc : \n
    1. Category Name
        a) Sub-Category1 Name: 20% patients
        b) Sub-Category2 Name: 10% patients
    \n

    For Example : \n

    1. Cost
        a) out of pocket expense of $2k / month: 10% patients
        b) Part D plan do not qualify for copay program: 10% patients
        c) not covered by their insurance: 10% patients
        d) very expensive: 10% patients
    """

    answer8 = process(answer7, Question8)




    Question9 = """
    Please write this in breif without loosing any information for each category. Categories and Sub-Categories are provided to you along with prcentages.
    """

    answer9 = process(answer8, Question9)

    RealQuestionA = """
    You are a Pharma Data Analyst. You have answer the below question asked by the leadership based on the data provided.
    You can mention personal details like Unique ID to refer patient.
    You should read the data very carefully and do a double check before answering.
    Use the instance where name of switched medication is mentioned and also name the swtiched brand. if any.
    Question : Have there been any instances where patients had to discontinue XYZ as suggested by HCP? If yes, why?
    """
    QuestionA = 'Have there been any instances where patients had to discontinue XYZ due to severe side effects and swtich to another brand?'
    AnswerA = process(csv_text,RealQuestionA)

    QuestionB = 'What is the relation between dosage levels and reported side effects?'
    AnswerB = process(csv_text,QuestionB)

    QuestionC = 'What are the primary reasons patients are taking XYZ?'
    AnswerC = process(csv_text,QuestionC)

    questions = []
    answers = []

    questions, answers = extract_questions_and_answers(str(answer6))

    print(questions)
    print(answers)



    QuestionD = 'questions[3]'
    AnswerD = 'answers[3]'
    QuestionE = 'questions[4]'
    AnswerE = 'answers[4]'

    plt_data = str(answer3)

    image_path, bar_colors = percentage_bar_charts(plt_data)

    combined_image_path = create_percentage_bar_charts(str(answer8),bar_colors)

    return answer_Alter, #image_path, combined_image_path list1,


def process_csv_with(upload):
    # Read the uploaded CSV file into a Pandas DataFrame
    data = pd.read_csv(upload)
    data_head = data.head()
    csv_text = data_head.to_csv(index=False)

    outputs = []

    df = pd.DataFrame()

    column_name = process(csv_text, relevent_column)

    column_name = str(column_name).strip()

    df = data[[column_name]]

    # For reading file name and type

    # format_of_file_location = upload.name
    # file_name = format_of_file_location.split("/")
    # name = file_name[-1]
    # str1 = f"{name} has been successfully uploaded"

    # Add a 'patient_id' column with row numbers
    df['patient_id'] = df.index + 1

    # Move the 'patient_id' column to be the first column
    columns1 = ['patient_id'] + [col for col in df.columns if col != 'patient_id']
    df = df[columns1]

    # Randomly sample 10 rows without replacement
    # df = df.sample(n=10)

    df = df.head(10)

    print(df.head())
    # Calculate the total number of rows using the shape attribute
    total_patients = df.shape[0]

    print(total_patients)

    # # Split the DataFrame into smaller DataFrames with 10 rows each
    # chunk_size = 10
    # num_chunks = len(data) // chunk_size
    # remainder = len(data) % chunk_size

    # dataframes = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    # if remainder > 0:
    #     dataframes.append(data.iloc[num_chunks*chunk_size:])

    csv_text = df.to_csv(index=False)

    Question1 = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    # Question1 = """
    # Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    # Provide the name of each reason and the number of patients who fall into that category, in descending order.
    # """

    # Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answer = process(csv_text, Question1) + " "

    print(answer)

    Question1_Alter = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Who are the top 2 patients who either already switched their medication or are most likely to switch or discontinue their medication?
    For Example one patient mentioned 'their doctor decided to switch the patient's medication to Brand B'
    Also mention, What might be the probable reasons for this. While refereing any patient e.g. Patient 6 add Prefix "P" means "Patient 6" would be denoted as "Patient P6". If you are refering
    only one patient like P6 in response please consider it as singular dont use words like them and thier for such cases.
    """

    answer_Alter = process(csv_text, Question1_Alter)

    QuestionX = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group with patient id.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    # Question1 = """
    # Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    # Provide the name of each reason and the number of patients who fall into that category, in descending order.
    # """

    # Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answerX = process(csv_text, QuestionX) + " "

    print(answerX)

    Question2 = "What is the name of all category in this data? provide answer in bullet points"
    answer2 = process(answer, Question2)

    # answer2 = plot_categories_in_boxes(str(answer2))

    Question3 = "Please calculate the percentage of patients for each reason and replace the number of patients with their respective percentages.Total number of patients are : " + str(
        total_patients)
    answer3 = process(answer, Question3)

    if "0%" in str(answer3):
        answer3 = str(answer3).replace(", Discontinuation: 0%.", ".")

    print("answer3" + str(answer3))

    Question5 = """For each category mentioned below what all brief insights mentioned in the overall data? \n Categories : \n""" + str(
        answer) + """.
    Your insights should be in subcategories format and should be unique.
    I am providing you with on sample example. Your answer should be in similiar format. Do not take data from below example, only take the format information.
    For Example : \n
    1. Cost
        a) out of pocket expense of $2k / month: Number of patients - 1
        b) Part D plan do not qualify for copay program: Number of patients - 1
        c) not covered by their insurance: Number of patients - 1
        d) very expensive: Number of patients - 3

    2. Side effects
        a) Nausea -4
        b) Bone Pain -2
        c) Abdominal bloating -1
        d) Body pain - 1
        e) Headache - 1

    \n

    Remember: You should also provide number of patient that fit into respective subcategory based on the data.
    """
    answer5 = process(csv_text, Question5)

    print('answer5 :' + str(answer5))

    Question6 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the ‘""" + column_name + """’ column, List down the 3 important questions to understand the insights along with their answers,
    excluding information about """ + answer2 + """ .\n
    The questions should aim to uncover patterns and trends in the data that would be useful for the Data Analysis team.?
    Your answers should be in below format (Only use the format, do not copy from this):\n
    1. Your question will come here? Answer: Your answer will come here.
    """
    answer6 = process(csv_text, Question6)

    Question7 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the Sub-Categories mentioned below,
    please carefully find the number of patients for each subcategory based on the data provided.
    You should double check your answer. Your answer should be correct.
    Categories and Sub-Categories : """ + answer5 + """. \n
    Do not mention patient_id in your answer just number of patients. \n
    Please do not miss any sub-categories. Just do a double check if you have included all the sub-categories. \n
    Your answers should be in below format (Only use the format, do not copy from this)):\n
    1. Cost
        a) out of pocket expense of $2k / month : Number of patients - 1
        b) Part D plan do not qualify for copay program : Number of patients - 1
        c) not covered by their insurance : Number of patients - 1
        d) very expensive : Number of patients - 1

    """
    answer7 = process(csv_text, Question7)

    print(answer7)

    Question8 = """
    You are a data analyst. You have been provided with some categories and sub-categories and their respective number of patients.
    Please calculate the percentage of patients for each Sub-Category and
    replace the number of patients with their respective percentages.Total number of patients are : """ + str(
        total_patients) + """.
    To caluclate the percentages you need to first divide number of patients written with Total number of patients provided to you.
    After that multiple them with 100 and add a '%' sign.
    Your final answer should be in the exact below format, I am also giving you an example. Please double check the format, the whitespaces etc : \n
    1. Category Name
        a) Sub-Category1 Name: 20% patients
        b) Sub-Category2 Name: 10% patients
    \n

    For Example : \n

    1. Cost
        a) out of pocket expense of $2k / month: 10% patients
        b) Part D plan do not qualify for copay program: 10% patients
        c) not covered by their insurance: 10% patients
        d) very expensive: 10% patients
    """

    answer8 = process(answer7, Question8)

    Question9 = """
    Please write this in breif without loosing any information for each category. Categories and Sub-Categories are provided to you along with prcentages.
    """

    answer9 = process(answer8, Question9)

    RealQuestionA = """
    You are a Pharma Data Analyst. You have answer the below question asked by the leadership based on the data provided.
    You can mention personal details like Unique ID to refer patient.
    You should read the data very carefully and do a double check before answering.
    Use the instance where name of switched medication is mentioned and also name the swtiched brand. if any.
    Question : Have there been any instances where patients had to discontinue XYZ as suggested by HCP? If yes, why?
    """
    QuestionA = 'Have there been any instances where patients had to discontinue XYZ due to severe side effects and swtich to another brand?'
    AnswerA = process(csv_text, RealQuestionA)

    QuestionB = 'What is the relation between dosage levels and reported side effects?'
    AnswerB = process(csv_text, QuestionB)

    QuestionC = 'What are the primary reasons patients are taking XYZ?'
    AnswerC = process(csv_text, QuestionC)

    questions = []
    answers = []

    questions, answers = extract_questions_and_answers(str(answer6))

    print(questions)
    print(answers)

    QuestionD = 'questions[3]'
    AnswerD = 'answers[3]'
    QuestionE = 'questions[4]'
    AnswerE = 'answers[4]'

    plt_data = str(answer3)

    image_path, bar_colors = percentage_bar_charts(plt_data)

    combined_image_path = create_percentage_bar_charts(str(answer8), bar_colors)

    return image_path


def process_columns(upload):
    # Read the uploaded CSV file into a Pandas DataFrame
    data = pd.read_csv(upload)
    data_head = data.head()
    csv_text = data_head.to_csv(index=False)

    outputs = []

    df = pd.DataFrame()

    column_name = process(csv_text, relevent_column)

    column_name = str(column_name).strip()

    df = data[[column_name]]

    # For reading file name and type

    # format_of_file_location = upload.name
    # file_name = format_of_file_location.split("/")
    # name = file_name[-1]
    # str1 = f"{name} has been successfully uploaded"

    # Add a 'patient_id' column with row numbers
    df['patient_id'] = df.index + 1

    # Move the 'patient_id' column to be the first column
    columns1 = ['patient_id'] + [col for col in df.columns if col != 'patient_id']
    df = df[columns1]

    # Randomly sample 10 rows without replacement
    # df = df.sample(n=10)

    df = df.head(10)

    print(df.head())
    # Calculate the total number of rows using the shape attribute
    total_patients = df.shape[0]

    print(total_patients)

    # # Split the DataFrame into smaller DataFrames with 10 rows each
    # chunk_size = 10
    # num_chunks = len(data) // chunk_size
    # remainder = len(data) % chunk_size

    # dataframes = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    # if remainder > 0:
    #     dataframes.append(data.iloc[num_chunks*chunk_size:])

    csv_text = df.to_csv(index=False)

    Question1 = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    # Question1 = """
    # Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    # Provide the name of each reason and the number of patients who fall into that category, in descending order.
    # """

    # Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answer = process(csv_text, Question1) + " "

    print(answer)

    Question1_Alter = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Who are the top 2 patients who either already switched their medication or are most likely to switch or discontinue their medication?
    For Example one patient mentioned 'their doctor decided to switch the patient's medication to Brand B'
    Also mention, What might be the probable reasons for this. While refereing any patient e.g. Patient 6 add Prefix "P" means "Patient 6" would be denoted as "Patient P6". If you are refering
    only one patient like P6 in response please consider it as singular dont use words like them and thier for such cases.
    """

    answer_Alter = process(csv_text, Question1_Alter)

    QuestionX = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group with patient id.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    # Question1 = """
    # Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    # Provide the name of each reason and the number of patients who fall into that category, in descending order.
    # """

    # Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answerX = process(csv_text, QuestionX) + " "

    print(answerX)

    Question2 = "What is the name of all category in this data? provide answer in bullet points"
    answer2 = process(answer, Question2)

    # answer2 = plot_categories_in_boxes(str(answer2))

    Question3 = "Please calculate the percentage of patients for each reason and replace the number of patients with their respective percentages.Total number of patients are : " + str(
        total_patients)
    answer3 = process(answer, Question3)

    if "0%" in str(answer3):
        answer3 = str(answer3).replace(", Discontinuation: 0%.", ".")

    print("answer3" + str(answer3))

    Question5 = """For each category mentioned below what all brief insights mentioned in the overall data? \n Categories : \n""" + str(
        answer) + """.
    Your insights should be in subcategories format and should be unique.
    I am providing you with on sample example. Your answer should be in similiar format. Do not take data from below example, only take the format information.
    For Example : \n
    1. Cost
        a) out of pocket expense of $2k / month: Number of patients - 1
        b) Part D plan do not qualify for copay program: Number of patients - 1
        c) not covered by their insurance: Number of patients - 1
        d) very expensive: Number of patients - 3

    2. Side effects
        a) Nausea -4
        b) Bone Pain -2
        c) Abdominal bloating -1
        d) Body pain - 1
        e) Headache - 1

    \n

    Remember: You should also provide number of patient that fit into respective subcategory based on the data.
    """
    answer5 = process(csv_text, Question5)

    print('answer5 :' + str(answer5))

    Question6 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the ‘""" + column_name + """’ column, List down the 3 important questions to understand the insights along with their answers,
    excluding information about """ + answer2 + """ .\n
    The questions should aim to uncover patterns and trends in the data that would be useful for the Data Analysis team.?
    Your answers should be in below format (Only use the format, do not copy from this):\n
    1. Your question will come here? Answer: Your answer will come here.
    """
    answer6 = process(csv_text, Question6)

    Question7 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the Sub-Categories mentioned below,
    please carefully find the number of patients for each subcategory based on the data provided.
    You should double check your answer. Your answer should be correct.
    Categories and Sub-Categories : """ + answer5 + """. \n
    Do not mention patient_id in your answer just number of patients. \n
    Please do not miss any sub-categories. Just do a double check if you have included all the sub-categories. \n
    Your answers should be in below format (Only use the format, do not copy from this)):\n
    1. Cost
        a) out of pocket expense of $2k / month : Number of patients - 1
        b) Part D plan do not qualify for copay program : Number of patients - 1
        c) not covered by their insurance : Number of patients - 1
        d) very expensive : Number of patients - 1

    """
    answer7 = process(csv_text, Question7)

    print(answer7)

    Question8 = """
    You are a data analyst. You have been provided with some categories and sub-categories and their respective number of patients.
    Please calculate the percentage of patients for each Sub-Category and
    replace the number of patients with their respective percentages.Total number of patients are : """ + str(
        total_patients) + """.
    To caluclate the percentages you need to first divide number of patients written with Total number of patients provided to you.
    After that multiple them with 100 and add a '%' sign.
    Your final answer should be in the exact below format, I am also giving you an example. Please double check the format, the whitespaces etc : \n
    1. Category Name
        a) Sub-Category1 Name: 20% patients
        b) Sub-Category2 Name: 10% patients
    \n

    For Example : \n

    1. Cost
        a) out of pocket expense of $2k / month: 10% patients
        b) Part D plan do not qualify for copay program: 10% patients
        c) not covered by their insurance: 10% patients
        d) very expensive: 10% patients
    """

    answer8 = process(answer7, Question8)

    Question9 = """
    Please write this in breif without loosing any information for each category. Categories and Sub-Categories are provided to you along with prcentages.
    """

    answer9 = process(answer8, Question9)

    RealQuestionA = """
    You are a Pharma Data Analyst. You have answer the below question asked by the leadership based on the data provided.
    You can mention personal details like Unique ID to refer patient.
    You should read the data very carefully and do a double check before answering.
    Use the instance where name of switched medication is mentioned and also name the swtiched brand. if any.
    Question : Have there been any instances where patients had to discontinue XYZ as suggested by HCP? If yes, why?
    """
    QuestionA = 'Have there been any instances where patients had to discontinue XYZ due to severe side effects and swtich to another brand?'
    AnswerA = process(csv_text, RealQuestionA)

    QuestionB = 'What is the relation between dosage levels and reported side effects?'
    AnswerB = process(csv_text, QuestionB)

    QuestionC = 'What are the primary reasons patients are taking XYZ?'
    AnswerC = process(csv_text, QuestionC)

    questions = []
    answers = []

    questions, answers = extract_questions_and_answers(str(answer6))

    print(questions)
    print(answers)

    QuestionD = 'questions[3]'
    AnswerD = 'answers[3]'
    QuestionE = 'questions[4]'
    AnswerE = 'answers[4]'

    plt_data = str(answer3)

    image_path, bar_colors = percentage_bar_charts(plt_data)

    combined_image_path = create_percentage_bar_charts(str(answer8), bar_colors)

    return combined_image_path


def process_csv_with_chart(upload):

    # Read the uploaded CSV file into a Pandas DataFrame
    data = pd.read_csv(upload)
    data_head = data.head()
    csv_text = data_head.to_csv(index=False)

    outputs = []

    df = pd.DataFrame()

    column_name = process(csv_text, relevent_column)

    column_name = str(column_name).strip()

    df = data[[column_name]]

    #For reading file name and type

    # format_of_file_location = upload.name
    # file_name = format_of_file_location.split("/")
    # name = file_name[-1]
    # str1 = f"{name} has been successfully uploaded"


    # Add a 'patient_id' column with row numbers
    df['patient_id'] = df.index + 1

    # Move the 'patient_id' column to be the first column
    columns1 = ['patient_id'] + [col for col in df.columns if col != 'patient_id']
    df = df[columns1]

    # Randomly sample 10 rows without replacement
    #df = df.sample(n=10)

    df = df.head(10)

    print(df.head())
    # Calculate the total number of rows using the shape attribute
    total_patients = df.shape[0]

    print(total_patients)

    # # Split the DataFrame into smaller DataFrames with 10 rows each
    # chunk_size = 10
    # num_chunks = len(data) // chunk_size
    # remainder = len(data) % chunk_size

    # dataframes = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    # if remainder > 0:
    #     dataframes.append(data.iloc[num_chunks*chunk_size:])

    csv_text = df.to_csv(index=False)


    Question1 = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    #Question1 = """
    #Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    #Provide the name of each reason and the number of patients who fall into that category, in descending order.
    #"""

    #Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answer = process(csv_text, Question1) + " "

    print(answer)

    Question1_Alter = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Who are the top 2 patients who either already switched their medication or are most likely to switch or discontinue their medication?
    For Example one patient mentioned 'their doctor decided to switch the patient's medication to Brand B'
    Also mention, What might be the probable reasons for this. While refereing any patient e.g. Patient 6 add Prefix "P" means "Patient 6" would be denoted as "Patient P6". If you are refering
    only one patient like P6 in response please consider it as singular dont use words like them and thier for such cases.
    """

    answer_Alter = process(csv_text, Question1_Alter)

    QuestionX = """You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the column '""" + column_name + """' from the provided data list down leading indicators
    why patients might stop taking, switch, or discontinue their medication?
    Also, For each category, kindly provide the number of patients who fall into that group with patient id.
    Indicators name should be precise and understandable and should not have any adjectives or terms like 'Other'.
    Please note that there should not be any unnecessary Indicators.
    """

    #Question1 = """
    #Using the ‘"""+column_name+"""’ column from the data, identify 5 main reasons why patients might stop taking, switch, or discontinue their medication.
    #Provide the name of each reason and the number of patients who fall into that category, in descending order.
    #"""

    #Question1 = "Could you please analyze the provided reviews and identify the primary factors or indicators that lead to patient non-persistence, switching, or discontinuation of medications?"
    answerX = process(csv_text, QuestionX) + " "

    print(answerX)

    Question2 = "What is the name of all category in this data? provide answer in bullet points"
    answer2 = process(answer,Question2)

    #answer2 = plot_categories_in_boxes(str(answer2))

    Question3 = "Please calculate the percentage of patients for each reason and replace the number of patients with their respective percentages.Total number of patients are : " + str(total_patients)
    answer3 = process(answer, Question3)

    if "0%" in str(answer3):
      answer3 = str(answer3).replace(", Discontinuation: 0%.", ".")

    print("answer3" + str(answer3))


    Question5 = """For each category mentioned below what all brief insights mentioned in the overall data? \n Categories : \n"""  + str(answer) + """.
    Your insights should be in subcategories format and should be unique.
    I am providing you with on sample example. Your answer should be in similiar format. Do not take data from below example, only take the format information.
    For Example : \n
    1. Cost
        a) out of pocket expense of $2k / month: Number of patients - 1
        b) Part D plan do not qualify for copay program: Number of patients - 1
        c) not covered by their insurance: Number of patients - 1
        d) very expensive: Number of patients - 3

    2. Side effects
        a) Nausea -4
        b) Bone Pain -2
        c) Abdominal bloating -1
        d) Body pain - 1
        e) Headache - 1

    \n

    Remember: You should also provide number of patient that fit into respective subcategory based on the data.
    """
    answer5 = process(csv_text, Question5)

    print('answer5 :' + str(answer5))

    Question6 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the ‘"""+column_name+"""’ column, List down the 3 important questions to understand the insights along with their answers,
    excluding information about """ + answer2 + """ .\n
    The questions should aim to uncover patterns and trends in the data that would be useful for the Data Analysis team.?
    Your answers should be in below format (Only use the format, do not copy from this):\n
    1. Your question will come here? Answer: Your answer will come here.
    """
    answer6 = process(csv_text, Question6)

    Question7 = """
    You are an Pharma Data Analyst. Analyse the provided data carefully.
    Based on the Sub-Categories mentioned below,
    please carefully find the number of patients for each subcategory based on the data provided.
    You should double check your answer. Your answer should be correct.
    Categories and Sub-Categories : """+answer5+ """. \n
    Do not mention patient_id in your answer just number of patients. \n
    Please do not miss any sub-categories. Just do a double check if you have included all the sub-categories. \n
    Your answers should be in below format (Only use the format, do not copy from this)):\n
    1. Cost
        a) out of pocket expense of $2k / month : Number of patients - 1
        b) Part D plan do not qualify for copay program : Number of patients - 1
        c) not covered by their insurance : Number of patients - 1
        d) very expensive : Number of patients - 1

    """
    answer7 = process(csv_text, Question7)

    print(answer7)

    Question8 = """
    You are a data analyst. You have been provided with some categories and sub-categories and their respective number of patients.
    Please calculate the percentage of patients for each Sub-Category and
    replace the number of patients with their respective percentages.Total number of patients are : """ + str(total_patients) +""".
    To caluclate the percentages you need to first divide number of patients written with Total number of patients provided to you.
    After that multiple them with 100 and add a '%' sign.
    Your final answer should be in the exact below format, I am also giving you an example. Please double check the format, the whitespaces etc : \n
    1. Category Name
        a) Sub-Category1 Name: 20% patients
        b) Sub-Category2 Name: 10% patients
    \n

    For Example : \n

    1. Cost
        a) out of pocket expense of $2k / month: 10% patients
        b) Part D plan do not qualify for copay program: 10% patients
        c) not covered by their insurance: 10% patients
        d) very expensive: 10% patients
    """

    answer8 = process(answer7, Question8)




    Question9 = """
    Please write this in breif without loosing any information for each category. Categories and Sub-Categories are provided to you along with prcentages.
    """

    answer9 = process(answer8, Question9)

    RealQuestionA = """
    You are a Pharma Data Analyst. You have answer the below question asked by the leadership based on the data provided.
    You can mention personal details like Unique ID to refer patient.
    You should read the data very carefully and do a double check before answering.
    Use the instance where name of switched medication is mentioned and also name the swtiched brand. if any.
    Question : Have there been any instances where patients had to discontinue XYZ as suggested by HCP? If yes, why?
    """
    QuestionA = 'Have there been any instances where patients had to discontinue XYZ due to severe side effects and swtich to another brand?'
    AnswerA = process(csv_text,RealQuestionA)

    QuestionB = 'What is the relation between dosage levels and reported side effects?'
    AnswerB = process(csv_text,QuestionB)

    QuestionC = 'What are the primary reasons patients are taking XYZ?'
    AnswerC = process(csv_text,QuestionC)

    questions = []
    answers = []

    questions, answers = extract_questions_and_answers(str(answer6))

    print(questions)
    print(answers)



    QuestionD = 'questions[3]'
    AnswerD = 'answers[3]'
    QuestionE = 'questions[4]'
    AnswerE = 'answers[4]'

    plt_data = str(answer3)

    image_path, bar_colors = percentage_bar_charts(plt_data)

    combined_image_path = create_percentage_bar_chartss(str(answer8),bar_colors)

    return combined_image_path



def patientinsights1(data):
    question = "Does the patient mention any challenges or barriers he/she has faced in adhering to the treatment with brand XYZ in this case note?"
    answer = process(data, question)
    return answer

def patientinsights2(data):
    question = "Is there any indication in the case note that the patient has switched from brand XYZ to another medication, and what reasons or factors led to this decision?"
    answer = process(data, question)
    return answer

def patientinsights3(data):
    question = "Did the case note contain any success stories or positive testimonials from the patient about his/her experience with brand XYZ? What specific benifits or improvements are highlighted?"
    answer = process(data, question)
    return answer

def additionalques(data, ques):
    answer = process(data, ques)
    return answer


def data_frame(upload,Additional_Question):

    # Read the uploaded CSV file into a Pandas DataFrame
    data = pd.read_csv(upload.name)
    data_head = data.head()
    csv_text = data_head.to_csv(index=False)

    outputs = []

    df = pd.DataFrame()

    column_name = process(csv_text, relevent_column)

    column_name = str(column_name).strip()

    df = data[[column_name]]

    # Add a 'patient_id' column with row numbers
    df['patient_id'] = df.index + 1

    # Move the 'patient_id' column to be the first column
    columns1 = ['patient_id'] + [col for col in df.columns if col != 'patient_id']
    df = df[columns1]

    # Randomly sample 10 rows without replacement
    #df = df.sample(n=10)

    df = df.head(10)


    return df


