# ***TextWave* Document Retrieval System**

TextWave is a Flask-based web service that generates relevant document snippets in response to a user query by retrieving and reranking candidate passages using configurable methods like TF-IDF and bag-of-words. It exposes a simple HTTP API for querying and supports modular reranker selection for flexible information retrieval.

## **Local Setup**

Keep in mind that the following requires Python 3.11 and `pip` by default. First, create and activate a virtual environment (if not already done so):

```
python3 -m venv venv

source venv/bin/activate
```

On Windows, the command is changed to `venv\Scripts\activate`., Next, install the necessary dependencies and allowed libraries, located within the `requirements.txt` file:

```
pip install -r requirements.txt
```

## **Running the Program Locally**

This program can be run both locally, and using Docker. Before working through these simple steps, ensure that you are located in the `textwave` folder (for all terminal windows used). We'll start by going over the steps needed to run it locally using Flask. First, run the Flask app:

```
python app.py
```

The Flask server will be available at http://127.0.0.1:5000.

Then, test the endpoint and Flask API with the curl command in a separate terminal window:

```
curl -X POST http://localhost:5000/generate \
-H "Content-Type: application/json" \
-d '{"query": "What is the role of antioxidants in green tea?"}'
```

## **Running the Program with Docker**

For the following steps, ensure you have Docker installed. Then, build and run the container while in the `{ROOTDIR}` folder:

```
docker build -f textwave/Dockerfile -t textwave-app .

docker run -p 5000:5000 textwave-app
```

The Flask server will again be available at http://127.0.0.1:5000.

Afterwards, in a separate terminal window, run the same curl command:

```
curl -X POST http://localhost:5000/generate \
-H "Content-Type: application/json" \
-d '{"query": "What is the role of antioxidants in green tea?"}'
```
