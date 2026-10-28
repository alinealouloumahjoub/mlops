import requests  # Send HTTP requests from Python


# ===== 1. Define the API URL =====
URL = "http://localhost:8000"


# ===== 2. Check whether the API is running =====
response = requests.get(URL + "/health")

print("GET /health ->", response.json())


# ===== 3. Test with a customer who has low usage and low seniority =====
customer = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10,
}

# Send the customer data as JSON to the /predict endpoint.
response = requests.post(
    URL + "/predict",
    json=customer
)

print(
    "POST /predict (customer 1) ->",
    response.status_code,
    response.json()
)


# ===== 4. Test with a loyal customer =====
# This customer has high usage and high seniority.
customer2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80,
}

response = requests.post(
    URL + "/predict",
    json=customer2
)

print(
    "POST /predict (customer 2) ->",
    response.status_code,
    response.json()
)


# ===== 5. Test input validation with an invalid age =====
# Create a copy of customer with age changed to "abc".
bad = dict(customer, age="abc")

response = requests.post(
    URL + "/predict",
    json=bad
)

print(
    "POST /predict (age='abc') ->",
    response.status_code
)

print(
    "Message:",
    response.json()["detail"][0]["msg"]
)