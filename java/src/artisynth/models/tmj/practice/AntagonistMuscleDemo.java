package artisynth.models.tmj.practice;

import java.awt.Color;
import java.io.IOException;
import java.util.Random;

import artisynth.core.gui.ControlPanel;
import artisynth.core.materials.SimpleAxialMuscle;
import artisynth.core.mechmodels.FrameMarker;
import artisynth.core.mechmodels.MechModel;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.mechmodels.Particle;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.modelbase.ModelComponent;
import artisynth.core.probes.NumericOutputProbe;
import artisynth.core.workspace.RootModel;
import maspack.matrix.RigidTransform3d;
import maspack.render.RenderProps;

/**
 * Simple antagonist-muscle demo for the ArtiSynth/Python RL prototype.
 * A rigid body carries the tracked marker, and two opposing muscles pull it
 * toward a randomized target.
 */
public class AntagonistMuscleDemo extends RootModel {

   MechModel mech;
   Particle leftAnchor;
   Particle rightAnchor;
   Particle target;
   RigidBody box;
   FrameMarker marker;
   Muscle leftMuscle;
   Muscle rightMuscle;
   MuscleExciter leftExciter;
   MuscleExciter rightExciter;
   AntagonistMuscleRlController rlController;

   RlRestApi restApi;
   int restPort = 8081;

   Random random = new Random();
   double targetMinX = -0.25;
   double targetMaxX = 0.25;
   boolean testMode = false;

   public void build (String[] args) throws IOException {

      mech = new MechModel ("mech");
      addModel (mech);

      mech.setGravity (0, 0, 0);
      mech.setFrameDamping (1.0);
      mech.setRotaryDamping (0.5);

      leftAnchor = new Particle ("leftAnchor", 2, -0.60, 0, 0);
      leftAnchor.setDynamic (false);

      rightAnchor = new Particle ("rightAnchor", 2, 0.60, 0, 0);
      rightAnchor.setDynamic (false);

      box = RigidBody.createBox ("box", 0.5, 0.3, 0.3, 20);
      box.setPose (new RigidTransform3d (0.25, 0, 0));

      marker = new FrameMarker (-0.25, 0, 0);
      marker.setName ("marker");
      marker.setFrame (box);

      target = new Particle ("target", 1, 0.20, 0, 0);
      target.setDynamic (false);

      leftMuscle = new Muscle ("leftMuscle", 0);
      leftMuscle.setPoints (leftAnchor, marker);
      leftMuscle.setMaterial (
         new SimpleAxialMuscle (
            /*stiffness=*/0, /*damping=*/2, /*maxf=*/20));

      rightMuscle = new Muscle ("rightMuscle", 0);
      rightMuscle.setPoints (rightAnchor, marker);
      rightMuscle.setMaterial (
         new SimpleAxialMuscle (
            /*stiffness=*/0, /*damping=*/2, /*maxf=*/20));

      mech.addParticle (leftAnchor);
      mech.addParticle (rightAnchor);
      mech.addParticle (target);
      mech.addRigidBody (box);
      mech.addFrameMarker (marker);
      mech.addAxialSpring (leftMuscle);
      mech.addAxialSpring (rightMuscle);

      leftExciter = new MuscleExciter ("leftMuscleExciter");
      leftExciter.addTarget (leftMuscle, 1.0);
      mech.addMuscleExciter (leftExciter);

      rightExciter = new MuscleExciter ("rightMuscleExciter");
      rightExciter.addTarget (rightMuscle, 1.0);
      mech.addMuscleExciter (rightExciter);

      rlController = new AntagonistMuscleRlController (
         marker,
         target,
         new Muscle[] { leftMuscle, rightMuscle },
         new MuscleExciter[] { leftExciter, rightExciter });
      addController (rlController);

      addControlPanel();
      addOutputProbes();

      startRestServer();

      addTracingProbe (marker, "position", 0, 30);

      mech.setBounds (-1, -1, -1, 1, 1, 1);

      RenderProps.setSphericalPoints (leftAnchor, 0.06, Color.BLUE);
      RenderProps.setSphericalPoints (rightAnchor, 0.06, Color.CYAN);
      RenderProps.setSphericalPoints (marker, 0.06, Color.BLUE);
      RenderProps.setSphericalPoints (target, 0.07, Color.GREEN);
      RenderProps.setSpindleLines (leftMuscle, 0.02, Color.RED);
      RenderProps.setSpindleLines (rightMuscle, 0.02, Color.MAGENTA);
      RenderProps.setFaceColor (box, Color.LIGHT_GRAY);
   }

   private void addControlPanel() {
      ControlPanel panel = new ControlPanel ("controls");
      panel.addWidget (
         "left action", rlController, "leftActionExcitation", 0.0, 1.0);
      panel.addWidget (
         "right action", rlController, "rightActionExcitation", 0.0, 1.0);
      panel.addWidget ("left exciter", leftExciter, "excitation");
      panel.addWidget ("right exciter", rightExciter, "excitation");
      panel.addWidget ("left muscle excitation", leftMuscle, "excitation");
      panel.addWidget ("right muscle excitation", rightMuscle, "excitation");
      panel.addWidget (mech, "gravity");
      panel.addWidget (box, "mass");
      addControlPanel (panel);
   }

   private void addOutputProbes() {
      addProbe ("marker position", marker, "position");
      addProbe ("marker velocity", marker, "velocity");
      addProbe ("target position", target, "position");
      addProbe ("left exciter excitation", leftExciter, "excitation");
      addProbe ("right exciter excitation", rightExciter, "excitation");
      addProbe ("left muscle excitation", leftMuscle, "excitation");
      addProbe ("right muscle excitation", rightMuscle, "excitation");
      addProbe ("left muscle force", leftMuscle, "forceNorm");
      addProbe ("right muscle force", rightMuscle, "forceNorm");
   }

   private void addProbe (String name, ModelComponent comp, String propName) {
      try {
         NumericOutputProbe probe =
            new NumericOutputProbe (comp, propName, 0, 30, 0.01);
         probe.setName (name);
         addOutputProbe (probe);
      }
      catch (Exception e) {
         e.printStackTrace();
      }
   }

   private void startRestServer() {
      if (restApi != null) {
         return;
      }

      restApi = new RlRestApi (
         restPort,
         rlController,
         new RlRestApi.ModelHooks() {
            public void resetEpisode() {
               AntagonistMuscleDemo.this.resetEpisode();
            }

            public void setSeed (long seed) {
               random.setSeed (seed);
            }

            public void setTestMode (boolean value) {
               testMode = value;
            }
         });
      try {
         restApi.start();
         System.out.println (
            "AntagonistMuscleDemo REST API started on port " + restPort);
      }
      catch (IOException e) {
         restApi = null;
         System.err.println (
            "WARNING: AntagonistMuscleDemo loaded, but REST API could not "
            + "start on port " + restPort + ".");
         System.err.println (
            "Close any old ArtiSynth instance using this port, then reload "
            + "the model if Python cannot connect.");
         e.printStackTrace();
      }
   }

   private synchronized void resetEpisode() {
      rlController.setExcitations (new double[rlController.getActionSize()]);
      leftExciter.setExcitation (0.0);
      rightExciter.setExcitation (0.0);
      leftMuscle.setExcitation (0.0);
      rightMuscle.setExcitation (0.0);

      box.setPose (new RigidTransform3d (0.25, 0, 0));
      box.setVelocity (0, 0, 0, 0, 0, 0);

      target.setPosition (sampleTargetX(), 0, 0);
      target.setVelocity (0, 0, 0);
   }

   private double sampleTargetX() {
      double x = 0.0;
      while (Math.abs (x) < 0.05) {
         x = targetMinX + random.nextDouble() * (targetMaxX - targetMinX);
      }
      return x;
   }

}
