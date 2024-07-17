import requests

#TODO Test is not working for inverted_index.py yet!

URL = "http://localhost:5000/search"  

query = "information retrieval"

payload = {
    "query": query
}

response = requests.post(URL, json=payload)

print(f"Request sent to {URL}")
print(f"Query sent: {query}")
print(f"Response status code: {response.status_code}")
print(f"Response JSON: {response.json()}")