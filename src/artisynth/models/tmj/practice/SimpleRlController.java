package artisynth.models.tmj.practice;

import java.util.Locale;

import artisynth.core.mechmodels.FrameMarker;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.mechmodels.Point;
import artisynth.core.modelbase.ControllerBase;
import maspack.matrix.Point3d;
import maspack.matrix.Vector3d;
import maspack.properties.PropertyList;

public class SimpleRlController extends ControllerBase {

   public static PropertyList myProps =
      new PropertyList (SimpleRlController.class, ControllerBase.class);

   static {
      myProps.add ("leftActionExcitation", "left muscle action value", 0.0);
      myProps.add ("rightActionExcitation", "right muscle action value", 0.0);
   }

   public PropertyList getAllPropertyInfo() {
      return myProps;
   }

   private FrameMarker marker;
   private Point target;
   private Muscle leftMuscle;
   private Muscle rightMuscle;
   private MuscleExciter leftExciter;
   private MuscleExciter rightExciter;

   private double leftActionExcitation = 0.0;
   private double rightActionExcitation = 0.0;

   public SimpleRlController (
      FrameMarker marker,
      Point target,
      Muscle leftMuscle,
      Muscle rightMuscle,
      MuscleExciter leftExciter,
      MuscleExciter rightExciter) {

      this.marker = marker;
      this.target = target;
      this.leftMuscle = leftMuscle;
      this.rightMuscle = rightMuscle;
      this.leftExciter = leftExciter;
      this.rightExciter = rightExciter;
   }

   public synchronized double getLeftActionExcitation() {
      return leftActionExcitation;
   }

   public synchronized void setLeftActionExcitation (double value) {
      leftActionExcitation = clamp01 (value);
   }

   public synchronized double getRightActionExcitation() {
      return rightActionExcitation;
   }

   public synchronized void setRightActionExcitation (double value) {
      rightActionExcitation = clamp01 (value);
   }

   public synchronized void setExcitations (double[] actions) {
      if (actions == null || actions.length < 2) {
         throw new IllegalArgumentException ("Expected two excitation values");
      }
      leftActionExcitation = clamp01 (actions[0]);
      rightActionExcitation = clamp01 (actions[1]);
   }

   public synchronized double[] getExcitations() {
      return new double[] { leftActionExcitation, rightActionExcitation };
   }

   public int getActionSize() {
      return 2;
   }

   public int getStateSize() {
      return getState().length;
   }

   public int getObservationSize() {
      return getObservation().length;
   }

   public void apply (double t0, double t1) {
      double[] actions = getExcitations();
      leftExciter.setExcitation (actions[0]);
      rightExciter.setExcitation (actions[1]);
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
      Point3d pos = marker.getPosition();
      Point3d targetPos = target.getPosition();
      Vector3d vel = marker.getVelocity();
      double error = getTrackingError();
      double[] actions = getExcitations();

      return new double[] {
         pos.x, pos.y, pos.z,
         vel.x, vel.y, vel.z,
         targetPos.x, targetPos.y, targetPos.z,
         error,
         actions[0], actions[1],
         leftExciter.getExcitation(), rightExciter.getExcitation(),
         leftMuscle.getExcitation(), rightMuscle.getExcitation(),
         leftMuscle.getForceNorm(), rightMuscle.getForceNorm(),
      };
   }

   public String getExcitationsJson() {
      double[] actions = getExcitations();
      return String.format (Locale.US, "[%.6f,%.6f]", actions[0], actions[1]);
   }

   public String getStateJson() {
      Point3d pos = marker.getPosition();
      Point3d targetPos = target.getPosition();
      Vector3d vel = marker.getVelocity();
      double error = getTrackingError();
      double[] actions = getExcitations();

      return String.format (
         Locale.US,
         "{"
         + "\"markerPosition\":[%.6f,%.6f,%.6f],"
         + "\"markerVelocity\":[%.6f,%.6f,%.6f],"
         + "\"targetPosition\":[%.6f,%.6f,%.6f],"
         + "\"trackingError\":%.6f,"
         + "\"actionExcitations\":[%.6f,%.6f],"
         + "\"exciterExcitations\":[%.6f,%.6f],"
         + "\"muscleExcitations\":[%.6f,%.6f],"
         + "\"muscleForces\":[%.6f,%.6f],"
         + "\"rewardLike\":%.6f"
         + "}",
         pos.x, pos.y, pos.z,
         vel.x, vel.y, vel.z,
         targetPos.x, targetPos.y, targetPos.z,
         error,
         actions[0], actions[1],
         leftExciter.getExcitation(), rightExciter.getExcitation(),
         leftMuscle.getExcitation(), rightMuscle.getExcitation(),
         leftMuscle.getForceNorm(), rightMuscle.getForceNorm(),
         getRewardLikeValue());
   }

   public double getTrackingError() {
      Point3d pos = marker.getPosition();
      Point3d targetPos = target.getPosition();
      double dx = pos.x - targetPos.x;
      double dy = pos.y - targetPos.y;
      double dz = pos.z - targetPos.z;
      return Math.sqrt (dx * dx + dy * dy + dz * dz);
   }

   public double getRewardLikeValue() {
      double error = getTrackingError();
      double effort =
         leftActionExcitation * leftActionExcitation
         + rightActionExcitation * rightActionExcitation;
      double coactivation = leftActionExcitation * rightActionExcitation;
      return -error * error - 0.01 * effort - 0.02 * coactivation;
   }

   private double clamp01 (double value) {
      return Math.max (0.0, Math.min (1.0, value));
   }
}
