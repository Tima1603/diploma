# views.py
from django.shortcuts import render
import pandas as pd
from .kmeans import K_Means
import numpy as np


def kmeans_view(request):

    return render(request, 'kmeans/kmeans.html')
