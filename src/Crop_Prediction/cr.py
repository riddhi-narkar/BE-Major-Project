import numpy as np
import pandas as pd
from flask import Flask, render_template, request, Markup
import config
import requests

import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
crop_recommendation_model_path = 'model/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))
city = input("Enter the City: ")
N = int(input("Enter the Nitrogen value: "))
P = int(input("Enter the Phosphoros value: "))
K = int(input("Enter the Pottasium value: "))
ph = float(input("Enter the pH value: "))
rainfall = float(input("Enter the amount of rainfall recieved: "))
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

if weather_fetch(city) != None:
    temperature, humidity = weather_fetch(city)
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]
    print(final_prediction)
else:
    print("Check if the entered city name was not mispelled")



