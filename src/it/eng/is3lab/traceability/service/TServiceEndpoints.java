/*
 * Traceability Service Endpoints
 * 
 * Author: Luigi di Corrado
 * Mail: luigi.dicorrado@eng.it
 * Date: 18/12/2020
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
 * http://localhost:9080/EstimateMilkQualityModule/v1/milkQualityTraining
 * 
 * Training with configuration URL example 
 * http://localhost:9080/EstimateMilkQualityModule/v1/milkQualityTraining/randomstate/15/estimators/100
 * 
 * Prediction complete URL example 
 * http://localhost:9080/EstimateMilkQualityModule/v1/milkQualityPrediction
 * 
 * 
 * Method      : training
 * 
 * Endpoint    : /v1/milkQualityTraining
 * 
 * Type        : GET
 * 
 *               KEY          | VALUE
 *               -------------|-----------------
 * Headers     : Content-Type | application/json
 * 		         Accept       | application/json
 *    
 * Description : Trigger Python random forest module to execute the training
 * 
 * Request     : 
 * 
 * Response    : AIM containing training results
 * 
 * 
 * 
 * Method      : prediction
 * 
 * Endpoint    : /v1/milkQualityPredictions
 * 
 * Type        : GET
 * 
 *               KEY          | VALUE
 *               -------------|-----------------
 * Headers     : Content-Type | application/json
 * 		         Accept       | application/json
 * 
 * Description : Trigger Python random forest module to execute the prediction
 * 
 * Request     : 
 * 
 * Response    : AIM containing prediction results
 * 
 * 
 * 
 * Method      : configAndSendDatasetTraining
 * 
 * Endpoint    : /v1/milkQualityTraining/randomstate/{int randomstate}/estimators/{int estimators} 
 * 
 * Type        : GET
 * 
 *               KEY          | VALUE
 *               -------------|-----------------
 * Headers     : Content-Type | application/json
 * 		         Accept       | application/json
 *    
 * Description : Configure random state and estimators,
 * 				 execute random forest training and send the output as response.
 * 
 * Request     : Json input data to be stored for future training
 * 
 * Response    : AIM containing training results
 */

package it.eng.is3lab.traceability.service;

import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
//import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jboss.resteasy.annotations.providers.jaxb.Formatted;
//import it.eng.is3lab.traceability.pyplugin.RFConfigurator;

@Path("/v1")
@Consumes(MediaType.APPLICATION_JSON+";charset=UTF-8")
@Produces(MediaType.APPLICATION_JSON+";charset=UTF-8")
public class TServiceEndpoints implements TService{
	private static final Logger log = LogManager.getLogger(TServiceEndpoints.class);

    @GET
    @Path("/milkQualityTraining")
    @Formatted
	public synchronized Response training() {
    	TResult result = new TResult();
    	try {
    		log.debug("Training endpoint reached!");   		
    		String jsonDataOutput = TDataManagement.sendToPythonAndGetResult("Training");
    		log.debug("Training dataset successfully retrieved!");
    		log.debug("==========================================================");
    		return Response.status(200).entity(jsonDataOutput).build();
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
			result.setErrorCode("1");
			result.setErrorDesc(e.toString());
			result.setResult(false);
			return Response.status(500).entity(result).build();
		}
	}

    @GET
    @Path("/milkQualityPrediction")
    @Formatted
	public synchronized Response prediction() {
    	TResult result = new TResult();
    	try {
    		log.debug("Prediction endpoint reached!");
    		String jsonDataOutput = TDataManagement.sendToPythonAndGetResult("Prediction");
    		log.debug("Prediction dataset successfully retrieved!");
    		log.debug("==========================================================");
    		return Response.status(200).entity(jsonDataOutput).build();
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
			result.setErrorCode("1");
			result.setErrorDesc(e.toString());
			result.setResult(false);
			return Response.status(500).entity(result).build();
		}
	}
    
    /* TEMPORARY OFFLINE...
    @GET
    @Path("/milkQualityTraining/randomstate/{randomstate}/estimators/{estimators}")
    @Formatted
	public Response configAndSendDatasetTraining(@PathParam("randomstate") int randomstate,	@PathParam("estimators") int estimators) {
    	TResult result = new TResult();
    	String outputData = "";
    	try {
    		log.debug("Config Send training dataset endpoint reached!");
    		RFConfigurator rfConf = new RFConfigurator();
    		rfConf.setConfiguration(randomstate,estimators);
    		outputData = TDataManagement.sendToPythonAndGetResult("Training");
    		log.debug("Send training dataset complete!");
    		log.debug("==========================================================");
    		return Response.status(200).entity("RESULT "+outputData).build();
		} catch (Exception e) {
			log.error("An exception occured!",e);
			e.printStackTrace();
			result.setErrorCode("1");
			result.setErrorDesc(e.toString());
			result.setResult(false);
			return Response.status(500).entity(result).build();
		}
	}*/
}
