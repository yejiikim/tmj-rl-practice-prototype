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
      myProps.add ("actionExcitation", "external action value", 0.0);
   }

   public PropertyList getAllPropertyInfo() {
      return myProps;
   }

   private FrameMarker marker;
   private Point target;
   private Muscle muscle;
   private MuscleExciter exciter;

   private double actionExcitation = 0.0;

   public SimpleRlController (
      FrameMarker marker, Point target, Muscle muscle, MuscleExciter exciter) {

      this.marker = marker;
      this.target = target;
      this.muscle = muscle;
      this.exciter = exciter;
   }

   public synchronized double getActionExcitation() {
      return actionExcitation;
   }

   public synchronized void setActionExcitation (double value) {
      actionExcitation = clamp01 (value);
   }

   public synchronized void setExcitations (double[] actions) {
      if (actions == null || actions.length < 1) {
         throw new IllegalArgumentException ("Expected one excitation value");
      }
      actionExcitation = clamp01 (actions[0]);
   }

   public synchronized double[] getExcitations() {
      return new double[] { actionExcitation };
   }

   public int getActionSize() {
      return 1;
   }

   public void apply (double t0, double t1) {
      exciter.setExcitation (getActionExcitation());
   }

   public double[] getState() {
      Point3d pos = marker.getPosition();
      Point3d targetPos = target.getPosition();
      Vector3d vel = marker.getVelocity();
      double error = getTrackingError();

      return new double[] {
         pos.x, pos.y, pos.z,
         vel.x, vel.y, vel.z,
         targetPos.x, targetPos.y, targetPos.z,
         error,
         getActionExcitation(),
         exciter.getExcitation(),
         muscle.getForceNorm(),
         getRewardLikeValue()
      };
   }

   public String getExcitationsJson() {
      return String.format (Locale.US, "[%.6f]", getActionExcitation());
   }

   public String getStateJson() {
      Point3d pos = marker.getPosition();
      Point3d targetPos = target.getPosition();
      Vector3d vel = marker.getVelocity();
      double error = getTrackingError();

      return String.format (
         Locale.US,
         "{"
         + "\"markerPosition\":[%.6f,%.6f,%.6f],"
         + "\"markerVelocity\":[%.6f,%.6f,%.6f],"
         + "\"targetPosition\":[%.6f,%.6f,%.6f],"
         + "\"trackingError\":%.6f,"
         + "\"actionExcitation\":%.6f,"
         + "\"exciterExcitation\":%.6f,"
         + "\"muscleExcitation\":%.6f,"
         + "\"muscleForce\":%.6f,"
         + "\"rewardLike\":%.6f"
         + "}",
         pos.x, pos.y, pos.z,
         vel.x, vel.y, vel.z,
         targetPos.x, targetPos.y, targetPos.z,
         error,
         getActionExcitation(),
         exciter.getExcitation(),
         muscle.getExcitation(),
         muscle.getForceNorm(),
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
      double effort = getActionExcitation();
      return -error * error - 0.01 * effort * effort;
   }

   private double clamp01 (double value) {
      return Math.max (0.0, Math.min (1.0, value));
   }
}
