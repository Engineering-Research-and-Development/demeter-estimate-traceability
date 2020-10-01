/*
 * Traceability Service Interface
 * 
 * Author: Luigi di Corrado
 * Mail: luigi.dicorrado@eng.it
 * Date: 21/09/2020
 * Company: Engineering Ingegneria Informatica S.p.A.
 */

package it.eng.is3lab.traceability.service;

import java.io.IOException;
import java.io.InputStream;

import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.Response;

public interface TService {
	
	public Response training();
	
	public Response prediction();
	
	public Response sendDatasetTraining(@Context HttpServletRequest request, InputStream requestBody) throws IOException;
	
	public Response sendDatasetPrediction(@Context HttpServletRequest request, InputStream requestBody) throws IOException;

}
