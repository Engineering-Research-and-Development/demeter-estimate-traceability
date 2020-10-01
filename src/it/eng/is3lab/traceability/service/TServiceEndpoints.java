/*
 * Traceability Service Endpoints
 * 
 * Author: Luigi di Corrado
 * Mail: luigi.dicorrado@eng.it
 * Date: 23/09/2020
 * Company: Engineering Ingegneria Informatica S.p.A.
 * 
 * Implements the Traceability Service interface and define the 
 * business service endpoints through the use of annotations.
 * 
 * The complete URL to call an endpoint is composed like:
 * http://SERVER_HOST/PROJECT_NAME/ENDPOINT_URL
 * 
 * The ENDPOINT_URL is composed by:
 * /Path value annotation of the class/Path value annotation of the method
 * 
 * Training complete URL example 
 * http://localhost:8080/EstimateMilkQualityModule/traceability/Training
 * 
 * Prediction complete URL example 
 * http://localhost:8080/EstimateMilkQualityModule/traceability/Predictions
 * 
 * 
 * Method      : training
 * 
 * Endpoint    : /traceability/Training 
 * 
 * Type        : POST
 * 
 *               KEY          | VALUE
 *               -------------|-----------------
 * Headers     : Content-Type | application/json
 * 		         Accept       | application/json
 *    
 * Description : Read the json string content of the body inside the request and send it
 * 				 as input to the Python module executor.
 * 				 After processing the data within random forest training module, 
 * 				 the response will send a json string as output with the processed data.
 * 
 * Request     : Json input data to be sent to random forest training module
 * 
 * Response    : Response containing json data output
 * 
 * 
 * 
 * Method      : prediction
 * 
 * Endpoint    : /traceability/Predictions
 * 
 * Type        : POST
 * 
 *               KEY          | VALUE
 *               -------------|-----------------
 * Headers     : Content-Type | application/json
 * 		         Accept       | application/json
 * 
 * Description : Read the json string content of the body inside the request and send it
 * 				 as input to the Python module executor.
 * 				 After processing the data within random forest prediction module, 
 * 				 the response will send a json string as output with the processed data.
 * 
 * Request     : Json input data to be sent to random forest prediction module
 * 
 * Response    : Response containing json data output
 * 
 * 
 * 
 * Method      : initDataAndSend 
 *    
 * Description : Initialize the request body data and send it to the Python module executor class.
 * 				 The operation string is used to select the task to perform:
 *               	- "Training" - Send data to training method
 *               	- "Prediction" - Send data to prediction method
 * 
 * Parameters  : InputStream requestBody
 * 				 String 	 operation
 * 
 * Return 	   : String jsonDataOutput
 */

package it.eng.is3lab.traceability.service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.jboss.resteasy.annotations.providers.jaxb.Formatted;

import it.eng.is3lab.traceability.pyplugin.PyModuleExecutor;

@Path("/v1")
@Consumes(MediaType.APPLICATION_JSON)
@Produces(MediaType.APPLICATION_JSON)
public class TServiceEndpoints implements TService{
	private static final Logger log = LogManager.getLogger(TServiceEndpoints.class);

    @GET
    @Path("/milkQualityTraining")
    @Formatted
	public Response training() {
    	try {
    		log.debug("Training endpoint reached!");
    		String jsonDataOutput = this.initDataAndSend("Training");
    		return Response.ok(jsonDataOutput).build();
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
			return Response.ok(e).build();
		}
	}

    @GET
    @Path("/milkQualityPrediction")
    @Formatted
	public Response prediction() {
    	try {
    		log.debug("Training endpoint reached!");
    		String jsonDataOutput = this.initDataAndSend("Prediction");
    		return Response.ok(jsonDataOutput).build();
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
			return Response.ok(e).build();
		}
	}
    
    @POST
    @Path("/milkQualityTraining")
    @Formatted
	public Response sendDatasetTraining(@Context HttpServletRequest request, InputStream requestBody) {
    	try {
    		log.debug("Send dataset training endpoint reached!");
    		//this.readDataAndStore(requestBody,"Training");
    		this.readDataAndSend(requestBody, "Training");
    		return Response.ok("Data received successfully!").build();
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
			return Response.ok(e).build();
		}
	}
    
    @POST
    @Path("/milkQualityPrediction")
    @Formatted
	public Response sendDatasetPrediction(@Context HttpServletRequest request, InputStream requestBody) {
    	try {
    		log.debug("Send dataset prediction endpoint reached!");
    		//this.readDataAndStore(requestBody,"Prediction");
    		this.readDataAndSend(requestBody, "Prediction");
    		return Response.ok("Data received successfully!").build();
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
			return Response.ok(e).build();
		}
	}
    
    private String initDataAndSend(String operation) {
    	log.debug("Initializing input data.");
    	String jsonDataOutput = "";
    	//String jsonDataInput = "";
    	//PyModuleExecutor pyExe = new PyModuleExecutor();
        log.debug("Initialization completed!");
    	try {
    		log.debug("Reading data...");
    		//jsonDataInput = TDataManagement.readFromFile(operation);
    		jsonDataOutput = TDataManagement.readFromFile(operation);
	        //log.debug("Sending data to python executor class.");
        	//jsonDataOutput = pyExe.executeFunction(jsonDataInput,operation);
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
		} 
		return jsonDataOutput;
    }
    
    private void readDataAndStore(InputStream requestBody,String operation) {
    	log.debug("Init reading data and store method...");
    	String line;
    	InputStreamReader inputStream = new InputStreamReader(requestBody);
		BufferedReader reader = new BufferedReader(inputStream);
        StringBuilder jsonDataInput = new StringBuilder();
        log.debug("Initialization completed!");
    	try {
    		log.debug("Reading request body.");
	        while ((line = reader.readLine()) != null) {
	        	jsonDataInput.append(line);
	        }
	        log.debug("Store dataset to file.");
	        TDataManagement.storeToFile(jsonDataInput.toString(),operation);	        
		} catch (IOException e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
		} finally {
    		if (reader != null) {
    			try {
    				log.debug("Closing the reader.");
    				reader.close();
    			}
    			catch (IOException e) {
    				log.error("An exception occured!",e);
    				e.printStackTrace();
    			}
    		}
    	}
    }
    
    private void readDataAndSend(InputStream requestBody,String operation) {
    	log.debug("Init reading data and store method...");
    	String jsonDataOutput = "";
    	String line;
    	PyModuleExecutor pyExe = new PyModuleExecutor();
    	InputStreamReader inputStream = new InputStreamReader(requestBody);
		BufferedReader reader = new BufferedReader(inputStream);
        StringBuilder jsonDataInput = new StringBuilder();
        log.debug("Initialization completed!");
    	try {
    		log.debug("Reading request body.");
	        while ((line = reader.readLine()) != null) {
	        	jsonDataInput.append(line);
	        }
	        //log.debug("Store dataset to file.");
	        //AWDataManagement.storeToFile(jsonDataInput.toString(),operation);
	        log.debug("Sending data to python executor class.");
        	jsonDataOutput = pyExe.executeFunction(jsonDataInput.toString(),operation);
        	log.debug("Store dataset to file.");
	        TDataManagement.storeToFile(jsonDataOutput,operation);
		} catch (IOException e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
		} finally {
    		if (reader != null) {
    			try {
    				log.debug("Closing the reader.");
    				reader.close();
    			}
    			catch (IOException e) {
    				log.error("An exception occured!",e);
    				e.printStackTrace();
    			}
    		}
    	}
    }

}
