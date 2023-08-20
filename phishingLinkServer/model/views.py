from django.shortcuts import render
from .phishing import test_url

def modelResult(url):
    booleanResult = test_url(url)
    print(booleanResult)
    print("result from Model : ",booleanResult)
    return booleanResult