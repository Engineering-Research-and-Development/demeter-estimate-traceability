![](https://portal.ogc.org/files/?artifact_id=92076)
# Estimate milk quality module

Machine learning module for traceability estimation using _**Random Forest**_ algorithm.
The values to predict are a discrete class label such as _**High**_, _**Medium**_ or _**Low**_ milk quality.
During training, it is necessary to provide to the model any historical data that is relevant 
to the problem domain and the true value we want the model learns to predict. 
The model learns any relationships between the data and the values we want to predict. 
The decision tree forms a structure, calculating the best questions to ask to make the most accurate estimates possible.

## Features

* **Algorithm Training, Testing and Metrics calculation:** 
Receives a dataset of _training features_ as input to perfom the training, test and metrics calculations. 
Return an object with the _training result_ that will show all the test records used after the training with the predictions made by the algorithm and the metrics.

* **Predict health condition:** 
Receive a dataset of _prediction features_ as input to perform predictions and return an object with the _prediction result_.  

# How to use this image

## Pull the image
	
`docker pull demeterengteam/estimate-milk-quality:candidate`

## Run the application

It's possible to run the application using `docker run` or `docker-compose`

### Docker run

`docker run -p 9280:8080 demeterengteam/estimate-milk-quality:candidate`

Set the preferred port to use instead of 9280.

### Docker-compose

Create a **docker-compose.yml** file into a folder.

*docker-compose.yml content:*

```
version: '3'

services:
    milkquality:
        image: demeterengteam/estimate-milk-quality:candidate
        ports:
          - '${HOST_PORT}:8080'
```

Create a **.env** file into the same folder of the above docker-compose:

*.env content:*
```
HOST_PORT=9280
```

Set **HOST_PORT** value to the preferred port to use instead of **9280**.

First run the command `docker-compose up` only for the very first time.

Then simply run `docker-compose start` to start the application and `docker-compose stop` to stop it.

Once started open any REST client (i.e. Postman) and send requests to the application endpoints.

### Endpoints

The base URL is composed like:
`http://[HOST]:[HOST_PORT]/EstimateMilkQualityModule/ENDPOINT`

Headers settings:

| Key          | Value            |
| :----------- | :--------------- |
| Content-Type | application/json |
| Accept       | application/json |

Endpoint informations:

| URL                         | Type     | Used for                                                             | Input                                  | Output                                                |
| :-------------------------- | :------: | :------------------------------------------------------------------- | :------------------------------------- | :---------------------------------------------------- |
| `/v1/milkQualityTraininig`  | **POST** | Train the algorithm, calculate the metrics and store the result data | Json data with actual health condition | A simple message with info about the process          |
| `/v1/milkQualityTraininig`  | **GET**  | Retrieve the training result data that was stored                    |                                        | Json with test predicted health condition and metrics |
| `/v1/milkQualityPrediction` | **POST** | Estimate the health condition and store the result data              | Json with data to be processed         | A simple message with info about the process          |
| `/v1/milkQualityPrediction` | **GET**  | Retrieve the prediction result data that was stored                  |                                        | Json with predicted health condition                  |

The `/v1/milkQualityTraininig` endpoint can be used also to change the **random state** and **estimators** parameters of the algorithm.
To accomplish that, just add the following path parameters to the URL:

* `/randomstate/value`

* `/estimators/value`	

Both values must be **integers** numbers.

For instance: 
**http://localhost:9280/EstimateMilkQualityModule/v1/milkQualityTraininig/randomstate/42/estimators/100**
This endpoint will first change the values of random state and estimators parameters and then execute the training.

## Important Notes

The application image don't contains any training data model preloaded, so the first request to execute
must be a training one. That will create the models needed for the prediction tasks.
Isn't necessary to execute another training on next use of the application except for the purpose of improve training accuracy.

## Source

[GitHub: Demeter Estimate Milk Quality][1] 

[1]: https://github.com/Engineering-Research-and-Development/demeter-estimate-traceability