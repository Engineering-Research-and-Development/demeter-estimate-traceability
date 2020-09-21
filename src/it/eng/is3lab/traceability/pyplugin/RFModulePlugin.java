/*
 * Random Forest Module Plugin
 * 
 * Author: Luigi di Corrado
 * Mail: luigi.dicorrado@eng.it
 * Date: 21/09/2020
 * Company: Engineering Ingegneria Informatica S.p.A.
 * 
 * Interface that will be used on PyModuleExecutor to proxify the calls
 * to functions inside the python modules for training and prediction.
 */

package it.eng.is3lab.traceability.pyplugin;

public interface RFModulePlugin {
	
	public String execRFTraining(String jsonData);
	
	public String execRFPrediction(String jsonData);
	
}
