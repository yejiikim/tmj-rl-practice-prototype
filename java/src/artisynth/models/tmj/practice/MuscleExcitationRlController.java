package artisynth.models.tmj.practice;

import java.util.Locale;

import artisynth.core.driver.Main;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.mechmodels.Point;
import artisynth.core.modelbase.ControllerBase;
import maspack.matrix.Point3d;
import maspack.matrix.Vector3d;

/**
 * Reusable controller for REST-driven muscle excitation.
 * It stores the latest action vector from Python and applies it to the
 * registered MuscleExciters during the ArtiSynth advance step.
 */
public class MuscleExcitationRlController extends ControllerBase
   implements RlController {

   private Point source;
   private Point target;
   private Muscle[] muscles;
   private MuscleExciter[] exciters;

   private double[] actionExcitations;

   public MuscleExcitationRlController (
      Point source,
      Point target,
      Muscle[] muscles,
      MuscleExciter[] exciters) {

      if (source == null || target == null) {
         throw new IllegalArgumentException (
            "Source and target points cannot be null");
      }
      if (muscles == null || exciters == null || muscles.length != exciters.length) {
         throw new IllegalArgumentException (
            "Muscle and exciter arrays must have the same length");
      }
      if (muscles.length == 0) {
         throw new IllegalArgumentException ("At least one muscle is required");
      }

      this.source = source;
      this.target = target;
      this.muscles = muscles;
      this.exciters = exciters;
      this.actionExcitations = new double[exciters.length];
   }

   public synchronized void setExcitations (double[] actions) {
      if (actions == null || actions.length < getActionSize()) {
         throw new IllegalArgumentException (
            "Expected " + getActionSize() + " excitation values");
      }
      for (int i = 0; i < getActionSize(); i++) {
         actionExcitations[i] = clamp01 (actions[i]);
      }
   }

   public synchronized double[] getExcitations() {
      double[] copy = new double[actionExcitations.length];
      for (int i = 0; i < actionExcitations.length; i++) {
         copy[i] = actionExcitations[i];
      }
      return copy;
   }

   public int getActionSize() {
      return exciters.length;
   }

   public int getStateSize() {
      return getState().length;
   }

   public int getObservationSize() {
      return getObservation().length;
   }

   public void apply (double t0, double t1) {
      double[] actions = getExcitations();
      for (int i = 0; i < exciters.length; i++) {
         exciters[i].setExcitation (actions[i]);
      }
   }

   public double[] getState() {
      double[] observation = getObservation();
      double[] state = new double[observation.length + 1];
      for (int i = 0; i < observation.length; i++) {
         state[i] = observation[i];
      }
      state[observation.length] = getRewardLikeValue();
      return state;
   }

   public double[] getObservation() {
      Point3d pos = source.getPosition();
      Point3d targetPos = target.getPosition();
      Vector3d vel = source.getVelocity();
      double error = getTrackingError();
      double[] actions = getExcitations();
      double[] exciterValues = getExciterValues();
      double[] muscleExcitations = getMuscleExcitationValues();
      double[] muscleForces = getMuscleForceValues();

      double[] observation = new double[10 + 4 * getActionSize()];
      int idx = 0;

      observation[idx++] = pos.x;
      observation[idx++] = pos.y;
      observation[idx++] = pos.z;
      observation[idx++] = vel.x;
      observation[idx++] = vel.y;
      observation[idx++] = vel.z;
      observation[idx++] = targetPos.x;
      observation[idx++] = targetPos.y;
      observation[idx++] = targetPos.z;
      observation[idx++] = error;
      idx = appendValues (observation, idx, actions);
      idx = appendValues (observation, idx, exciterValues);
      idx = appendValues (observation, idx, muscleExcitations);
      appendValues (observation, idx, muscleForces);

      return observation;
   }

   public String getExcitationsJson() {
      return vectorJson (getExcitations());
   }

   public String getStateJson() {
      Point3d pos = source.getPosition();
      Point3d targetPos = target.getPosition();
      Vector3d vel = source.getVelocity();
      Vector3d targetVel = target.getVelocity();
      double[] actions = getExcitations();
      double[] exciterValues = getExciterValues();
      double[] muscleExcitations = getMuscleExcitationValues();
      double[] muscleForces = getMuscleForceValues();

      return String.format (
         Locale.US,
         "{"
         + "\"time\":%.6f,"
         + "\"observation\":{"
         + "\"marker\":{\"position\":%s,\"velocity\":%s},"
         + "\"target\":{\"position\":%s,\"velocity\":%s}"
         + "},"
         + "\"excitations\":%s,"
         + "\"muscleForces\":%s,"
         + "\"properties\":[],"
         + "\"markerPosition\":%s,"
         + "\"markerVelocity\":%s,"
         + "\"targetPosition\":%s,"
         + "\"trackingError\":%.6f,"
         + "\"actionExcitations\":%s,"
         + "\"exciterExcitations\":%s,"
         + "\"muscleExcitations\":%s,"
         + "\"rewardLike\":%.6f"
         + "}",
         getCurrentTime(),
         pointJson (pos),
         vectorJson (vel),
         pointJson (targetPos),
         vectorJson (targetVel),
         vectorJson (actions),
         vectorJson (muscleForces),
         pointJson (pos),
         vectorJson (vel),
         pointJson (targetPos),
         getTrackingError(),
         vectorJson (actions),
         vectorJson (exciterValues),
         vectorJson (muscleExcitations),
         getRewardLikeValue());
   }

   public double getTrackingError() {
      Point3d pos = source.getPosition();
      Point3d targetPos = target.getPosition();
      double dx = pos.x - targetPos.x;
      double dy = pos.y - targetPos.y;
      double dz = pos.z - targetPos.z;
      return Math.sqrt (dx * dx + dy * dy + dz * dz);
   }

   public double getRewardLikeValue() {
      double error = getTrackingError();
      double effort = 0.0;
      double coactivation = 1.0;
      double[] actions = getExcitations();

      for (int i = 0; i < actions.length; i++) {
         effort += actions[i] * actions[i];
         coactivation *= actions[i];
      }
      if (actions.length < 2) {
         coactivation = 0.0;
      }
      return -error * error - 0.01 * effort - 0.02 * coactivation;
   }

   protected synchronized double getActionExcitation (int idx) {
      return actionExcitations[idx];
   }

   protected synchronized void setActionExcitation (int idx, double value) {
      actionExcitations[idx] = clamp01 (value);
   }

   private double[] getExciterValues() {
      double[] values = new double[exciters.length];
      for (int i = 0; i < exciters.length; i++) {
         values[i] = exciters[i].getExcitation();
      }
      return values;
   }

   private double[] getMuscleExcitationValues() {
      double[] values = new double[muscles.length];
      for (int i = 0; i < muscles.length; i++) {
         values[i] = muscles[i].getExcitation();
      }
      return values;
   }

   private double[] getMuscleForceValues() {
      double[] values = new double[muscles.length];
      for (int i = 0; i < muscles.length; i++) {
         values[i] = muscles[i].getForceNorm();
      }
      return values;
   }

   private int appendValues (double[] dest, int idx, double[] values) {
      for (int i = 0; i < values.length; i++) {
         dest[idx++] = values[i];
      }
      return idx;
   }

   private String pointJson (Point3d point) {
      return String.format (
         Locale.US, "[%.6f,%.6f,%.6f]", point.x, point.y, point.z);
   }

   private String vectorJson (Vector3d vector) {
      return String.format (
         Locale.US, "[%.6f,%.6f,%.6f]", vector.x, vector.y, vector.z);
   }

   private String vectorJson (double[] values) {
      StringBuilder sb = new StringBuilder();
      sb.append ("[");
      for (int i = 0; i < values.length; i++) {
         if (i > 0) {
            sb.append (",");
         }
         sb.append (String.format (Locale.US, "%.6f", values[i]));
      }
      sb.append ("]");
      return sb.toString();
   }

   private double getCurrentTime() {
      Main main = Main.getMain();
      return main == null ? 0.0 : main.getTime();
   }

   private double clamp01 (double value) {
      return Math.max (0.0, Math.min (1.0, value));
   }
}
