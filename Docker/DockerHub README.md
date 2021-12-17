<img src="https://portal.ogc.org/files/?artifact_id=92076" width="50%">

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

## Requirements

The following components are needed for start using this component:

| Components required                | Description   |
| :--------------------------------  | :------------ |
| [pilot4.2-traslator:candidate][2]  | AIM traslator |

## Pull the image
	
`docker pull demeterengteam/estimate-milk-quality:latest`

## Run the application

It's possible to run the application using <!--`docker run` or --> `docker-compose`

<!--
### Docker run

`docker run -p 9280:8080 demeterengteam/estimate-milk-quality:latest`

Set the preferred port to use instead of 9280.
-->

### Docker-compose

Create a **docker-compose.yml** file into a folder.

*docker-compose.yml content:*

```
version: '3'

services:
    milkquality:
        image: demeterengteam/estimate-milk-quality:latest
        ports:
          - '${HOST_PORT}:8080'
        environment: 
          - MQ_AIM_TRASLATOR_PREDICTION_URL=http://${HOST_IP}:${TRASLATOR_SERVICE_HOST_PORT}/demeter-csvManager/rest/traslator/v1/MilkQualityPrediction
          - MQ_AIM_TRASLATOR_TRAINING_URL=http://${HOST_IP}:${TRASLATOR_SERVICE_HOST_PORT}/demeter-csvManager/rest/traslator/v1/MilkQualityTraining    
```

Create a **.env** file into the same folder of the above docker-compose:

*.env content:*
```
HOST_PORT=9280
HOST_IP=**INSERT HOST IP**
TRASLATOR_SERVICE_HOST_PORT=**INSERT THE PORT OF THE TRASLATOR COMPONENT*
```

Set **HOST_PORT** value to the preferred port to use instead of **9280**.

Set **HOST_IP** value with the correct ip of the host machine.

Set **TRASLATOR_SERVICE_HOST_PORT** value equal to the port used for the traslator service component.

First run the command `docker-compose up` only for the very first time.

Then simply run `docker-compose start` to start the application and `docker-compose stop` to stop it.

Before doing any request, you must upload the data as csv file using the [traslator service][2].

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

| URL                         | Type    | Used for                                                            | Input | Output                                   |
| :-------------------------- | :-----: | :------------------------------------------------------------------ | :---- | :--------------------------------------- |
| `/v1/milkQualityTraininig`  | **GET** | Train the algorithm, calculate the metrics and send the result data |       | AIM Json output data result with metrics |
| `/v1/milkQualityPrediction` | **GET** | Estimate the health condition and send the result data              |       | AIM Json output data result              |
<!--
The `/v1/milkQualityTraininig` endpoint can be used also to change the **random state** and **estimators** parameters of the algorithm.
To accomplish that, just add the following path parameters to the URL:

* `/randomstate/value`

* `/estimators/value`	

Both values must be **integers** numbers.

For instance: 
**http://localhost:9280/EstimateMilkQualityModule/v1/milkQualityTraininig/randomstate/42/estimators/100**
This endpoint will first change the values of random state and estimators parameters and then execute the training.
-->

## Important Notes

The application image don't contains any training data model preloaded, so the first request to execute
must be a training one. That will create the models needed for the prediction tasks.
Isn't necessary to execute another training on next use of the application except for the purpose of improve training accuracy.

## Source

[GitHub: Demeter Estimate Milk Quality][1] 

[1]: https://github.com/Engineering-Research-and-Development/demeter-estimate-traceability
[2]: https://hub.docker.com/r/demeterengteam/pilot4.2-traslator

## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)