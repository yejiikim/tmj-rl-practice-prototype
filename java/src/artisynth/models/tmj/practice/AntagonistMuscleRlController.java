package artisynth.models.tmj.practice;

import artisynth.core.mechmodels.FrameMarker;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.mechmodels.Point;
import maspack.properties.PropertyList;

/**
 * Two-muscle convenience wrapper around MuscleExcitationRlController.
 */
public class AntagonistMuscleRlController extends MuscleExcitationRlController {

   public static PropertyList myProps =
      new PropertyList (
         AntagonistMuscleRlController.class,
         MuscleExcitationRlController.class);

   static {
      myProps.add ("leftActionExcitation", "left muscle action value", 0.0);
      myProps.add ("rightActionExcitation", "right muscle action value", 0.0);
   }

   public PropertyList getAllPropertyInfo() {
      return myProps;
   }

   public AntagonistMuscleRlController (
      FrameMarker marker,
      Point target,
      Muscle[] muscles,
      MuscleExciter[] exciters) {

      super (marker, target, muscles, exciters);
   }

   public synchronized double getLeftActionExcitation() {
      return getActionExcitation (0);
   }

   public synchronized void setLeftActionExcitation (double value) {
      setActionExcitation (0, value);
   }

   public synchronized double getRightActionExcitation() {
      if (getActionSize() < 2) {
         return 0.0;
      }
      return getActionExcitation (1);
   }

   public synchronized void setRightActionExcitation (double value) {
      if (getActionSize() >= 2) {
         setActionExcitation (1, value);
      }
   }
}
