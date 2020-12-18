/*
 * Traceability Service Interface
 * 
 * Author: Luigi di Corrado
 * Mail: luigi.dicorrado@eng.it
 * Date: 18/12/2020
 * Company: Engineering Ingegneria Informatica S.p.A.
 */

package it.eng.is3lab.traceability.service;

import javax.ws.rs.core.Response;

public interface TService {
	
	public Response training();
	
	public Response prediction();
	
	//public Response configAndSendDatasetTraining(int randomstate, int estimators);

}
