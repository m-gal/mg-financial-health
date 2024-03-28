"""
    Check API with FastAPI

    1. Run fast_api.py
    2. Run from Terminal in ./fhealth:
            >>> python api/fast_request.py
        or in ./fhealth/api:
            >>> python fast_request.py

    Response body must be:
    b'{"teu_esimated":561.0,"vu_esimated":0.4656}'

    @author: mikhail.galkin
"""

if __name__ == "__main__":
    import requests

    data = {
        "sic": 3420,
        "solvency_debt_ratio": 0.522054,
        "liquid_current_ratio": 6.336683,
        "liquid_quick_ratio": 2.558935,
        "liquid_cash_ratio": 0.334832,
        "profit_net_margin": 0.025725,
        "profit_roa": 0.029588,
        "active_acp": 72.378706,
    }
    response = requests.post("http://127.0.0.1:8000/finscore/predict", json=data)
    print(response.content)

"""
Interactive API docs:
Now to get the above result we had to manually call each endpoint but FastAPI
comes with Interactive API docs which can access by adding /docs in your path.
To access docs for our API we’ll go to http://127.0.0.1:8000/docs.
Here you’ll get the following page where you can test the endpoints of your API
by seeing the output they’ll give for the corresponding inputs if any.

Request body:
{
  "sic": 3420,
  "solvency_debt_ratio": 0.522054,
  "liquid_current_ratio": 6.336683,
  "liquid_quick_ratio": 2.558935,
  "liquid_cash_ratio": 0.334832,
  "profit_net_margin": 0.025725,
  "profit_roa": 0.029588,
  "active_acp": 72.378706
}

"""
