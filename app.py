from flask import Flask, render_template, request, redirect, url_for
from script import process_csv_with_columns
from script import process_csv_with
from script import process_columns
from script import process_csv_with_chart
from script import patientinsights1
from script import patientinsights2
from script import patientinsights3
from script import additionalques

app = Flask(__name__, static_url_path='/static')
text = ""  # Initialize text as a global variable

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Save the uploaded file
    # (Add your file saving logic here)

    # Return a success message
    return 'File uploaded successfully!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Upload')
def upload():
    return render_template('upload.html')

@app.route('/CaseNote Insight')
def casenote():
    return render_template('casenote.html')

@app.route('/Another Route')
def casenotee():
    return render_template('casenote.html')

@app.route('/Patient Insight')
def patient():
    return render_template('patient.html')

@app.route('/process', methods=['GET','POST'])
def process():
    if request.method == 'POST':
        file = request.files['file']
        #print(f"Received file: {file.filename}")
        file.save(file.filename)
        global output1
        output1 = process_csv_with_columns(file.filename)
        global output2
        output2 = process_csv_with(file.filename)
        global output3
        output3 = process_columns(file.filename)
        global output4
        output4 = process_csv_with_chart(file.filename)
    return render_template('casenote.html', answer1=output1, answer2=output2, answer3=output3, answer4 = output4)


@app.route('/add_data', methods=['GET','POST'])
def add_data():
    global text
    if request.method == 'POST':
        text = request.form.get('case_note')
        #ques = request.form.get('add_ques')
        global output11
        output11 = patientinsights1(text)

        global output12
        output12 = patientinsights2(text)

        global output13
        output13 = patientinsights3(text)

        #global output14
        #output14 = additionalques(text, ques)

    return render_template('patient.html', answer11=output11, answer12=output12, answer13=output13)

@app.route('/mp_data', methods=['GET','POST'])
def mp_data():
    global text
    if request.method == 'POST':
        ques = request.form.get('add_ques')
        global output
        output = additionalques(text, ques)
    return render_template('patient.html', answer = output)


# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
