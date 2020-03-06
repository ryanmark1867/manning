# manning
repo for Manning book **Deep Learning with Structured Data** https://www.manning.com/books/deep-learning-with-structured-data

## Note: 
You can find an improved repo for the book, including many updates to the code and a simpler structure, at https://github.com/ryanmark1867/deep_learning_for_structured_data

## Note about  the current state of the code and upcoming improvements
The code is being updated as the text of the book evolves. Here are some recommended improvements to the code that will be incorporated in future updates (post this review cycle):
- clean up variable names
- consider packaging with Docker to make exercising the code easier
- consider including a random forest treatment of the problem to contrast with the performance and complexity of the DL solution
- use Pandas chaining to make code cleaner
- use regex in cleanup of locations

## Prequisites
To run the code in this repo you will need access to an environment (such as Paperspace https://www.paperspace.com/ or Watson Studio Cloud https://cloud.ibm.com/catalog/services/watson-studio ) that supports Jupyter notebooks.

## Background

- **https://open.toronto.ca/dataset/ttc-streetcar-delay-data/** original dataset
- **https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl** Kaggle submission that was used as input to creation of the Keras model used in this example

## Directory structure
- **CSV_XLS** CSV and XLS data
- **pickled_pandas_dataframes** pickled versions on intermediate Pandas dataframes. See the **Intermediate datasets** section below for details
- **sql** files related to the section in Chapter 5 about how to perform standard SQL operations in Pandas
- root directory contains notebook files for the code examples in the book - see the **Code** section below for details

## Intermediate datasets (in pickled_pandas_dataframes directory)
- **2014_2018.pkl** pickled dataframe containing data from the original dataset 2014 to November 2018
- **2014_2018_df_cleaned_keep_bad_loc_geocoded_SCtrimmed_apr26.pkl** pickled dataframe with cleanup from streetcar_data_preparation.ipynb applied and latitude and longitude added for Location values (results of calling Geocoding API in streetcar_data_preparation-geocode.ipynb)

## Code
- **chapter2.ipynb** code snippets associated with introductory code in chapter 2
- **chapter5.ipynb** code snippets associated with SQL / Pandas examples
- **streetcar_data_preparation_refactored.ipynb** load original dataset into a single dataframe and perform basic cleanup on values. By default takes 2014_2019_upto_june.pkl as input
- **streetcar_data_exploration.ipynb** basic data exploration
- **streetcar_time_series.ipynb** additional data exploration using time series forecasting techniques
- **streetcar_data_preparation-geocode-public.ipynb** use Google Geocoding API to get latitude and longitude values from Location values. NOTE: to run the code in this file, you will need to get your own API Key from Google Cloud Platform https://developers.google.com/maps/documentation/embed/get-api-key
- **streetcar_DL_train-trimmed.ipynb** train Keras deep learning model. By default takes 2014_2018_df_cleaned_keep_bad_loc_geocoded_SCtrimmed_apr26.pkl as input
- **streetcar_DL_refactored.ipynb** train Keras deep learning model using refactored dataset. Create dataframe with entries for every date, hour, route, direction combination since Jan 1 2014. Join with dataframe containing the original dataset and generate target column that is 1 if there was a delay in that date / hour / route / direction combination and 0 otherwise. Use this refactored dataset to train Keras model. Focusing on this approach based on feedback received from presentation on this project at AI Beijing conference https://ai.oreilly.com.cn/ai-cn/public/schedule/detail/75461?locale=en
- **streetcar_data-geocode-get-boundaries.ipynb** code in development to get geo bounding boxes around each route so that DL_refactored can be augmented with subsets of the route - add a feature to that approach defined by the geographic subset of the route that the user is interested in.

**NOTE:** All code assumes that datasets (raw or pickled) are in a directory called "data" that is a sibling of the directory containing the notebook.  

## To re-run the entire flow from scratch:
1. copy all the XLS files from the original dataset (https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/#e8f359f0-2f47-3058-bf64-6ec488de52da) into a directory called "data" that is a sibling to the directory containing your notebook files
2. streetcar_data_preparation_refactored.ipynb with the following updates: 
- load_from_scratch set to  True
- pickled_output_dataframe set to the name you want for the output pickle file containing the cleansed dataframe
3. run streetcar_data_preparation-geocode-public.ipynb with the following updates:
- pickled_dataframe set to the pickle file you specified as output in step 2
- "gmaps = googlemaps.Client(key='')" update the key value to your own API key value https://developers.google.com/maps/documentation/embed/get-api-key See https://towardsdatascience.com/mapping-messy-addresses-part-1-getting-latitude-and-longitude-8fa7ba792430 for additional advice on using the Geocoding API, including how to increase your daily call limit so you can process the entire dataset in one run.
- file_outname set to the name you want for the output pickle file containing the cleansed dataframe including latitude and longitude columns
4. run streetcar_DL_train-trimmed.ipynb with the following updates:
- pickled_dataframe set to the pickle file you specified as output in step 3
