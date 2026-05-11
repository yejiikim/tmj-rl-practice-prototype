package artisynth.models.tmj.practice;

import artisynth.core.mechmodels.FrameMarker;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.mechmodels.Point;

/**
 * Compatibility name for the original two-muscle RL controller.
 */
public class SimpleRlController extends AntagonistMuscleRlController {

   public SimpleRlController (
      FrameMarker marker,
      Point target,
      Muscle[] muscles,
      MuscleExciter[] exciters) {

      super (marker, target, muscles, exciters);
   }
}
