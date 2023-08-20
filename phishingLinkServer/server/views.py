from django.shortcuts import render
from model.views import modelResult

# Create your views here.

def index(request):
    try:
     url = request.POST['urlToBeTested']
     resultFound = True
     boolResult = modelResult(url)
    except:
     resultFound = False
     boolResult = False
     url = ""
    
    return render(request,"index.html",{'resultFound':resultFound,'boolResult' : boolResult, 'url' : url})