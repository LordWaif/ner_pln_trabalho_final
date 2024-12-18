import argilla as rg
# from argilla.client import api
from dotenv import load_dotenv
load_dotenv()
import os
rg.Argilla(
    api_url=os.getenv('URL_ARGILLA'),
    api_key=os.getenv('USER_KEY'),
)