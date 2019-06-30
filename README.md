# manning
repo for Manning book Deep Learning with Structured Data

## Prequisites
To run the code in this repo you will need access to an environment (such as Paperspace https://www.paperspace.com/ or Watson Studio Cloud https://cloud.ibm.com/catalog/services/watson-studio ) that supports Jupyter notebooks.

## Background

- **AICH19_Using_deep_learning_and_time-series_forecasting_to_reduce_transit_delays.pdf** presentation material
- **https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/#e8f359f0-2f47-3058-bf64-6ec488de52da** original dataset
- **https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl** Kaggle submission that was used as input to creation of the Keras model used in this example

## Intermediate datasets
- **2014_2018.pkl** pickled dataframe containing data from the original dataset 2014 to November 2018
- **2014_2018_df_cleaned_keep_bad_loc_geocoded_SCtrimmed_apr26.pkl** pickled dataframe with cleanup from streetcar_data_preparation.ipynb applied and latitude and longitude added for Location values (results of calling Geocoding API in streetcar_data_preparation-geocode.ipynb)

## Code
- **streetcar_data_preparation.ipynb** load original dataset into a single dataframe and perform basic cleanup on values. By default takes 2014_2018.pkl as input
- **streetcar_data_exploration.ipynb** basic data exploration
- **streetcar_time_series.ipynb** additional data exploration using time series forecasting techniques
- **streetcar_data_preparation-geocode-public.ipynb** use Google Geocoding API to get latitude and longitude values from Location values. NOTE: to run the code in this file, you will need to get your own API Key from Google Cloud Platform https://developers.google.com/maps/documentation/embed/get-api-key
- **streetcar_DL_train-trimmed.ipynb** train Keras deep learning model. By default takes 2014_2018_df_cleaned_keep_bad_loc_geocoded_SCtrimmed_apr26.pkl as input

**NOTE:** All code assumes that datasets (raw or pickled) are in a directory called "data" that is a sibling of the directory containing the notebook.  

## To re-run the entire flow from scratch:
1. copy all the XLS files from the original dataset (https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/#e8f359f0-2f47-3058-bf64-6ec488de52da) into a directory called "data" that is a sibling to the directory containing your notebook files
2. run streetcar_data_preparation.ipynb with the following updates: 
- load_from_scratch set to  True
- pickled_output_dataframe set to the name you want for the output pickle file containing the cleansed dataframe
3. run streetcar_data_preparation-geocode-public.ipynb with the following updates:
- pickled_dataframe set to the pickle file you specified as output in step 2
- "gmaps = googlemaps.Client(key='')" update the key value to your own API key value https://developers.google.com/maps/documentation/embed/get-api-key See https://towardsdatascience.com/mapping-messy-addresses-part-1-getting-latitude-and-longitude-8fa7ba792430 for additional advice on using the Geocoding API, including how to increase your daily call limit so you can process the entire dataset in one run.
- file_outname set to the name you want for the output pickle file containing the cleansed dataframe including latitude and longitude columns
4. run streetcar_DL_train-trimmed.ipynb with the following updates:
- pickled_dataframe set to the pickle file you specified as output in step 3
