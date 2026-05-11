package artisynth.models.tmj.practice;

/**
 * Controller contract required by RlRestApi.
 */
public interface RlController {

   public int getActionSize();

   public int getStateSize();

   public int getObservationSize();

   public void setExcitations (double[] excitations);

   public String getExcitationsJson();

   public String getStateJson();
}
