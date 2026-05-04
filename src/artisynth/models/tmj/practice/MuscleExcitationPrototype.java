package artisynth.models.tmj.practice;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import artisynth.core.driver.Main;
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

public class MuscleExcitationPrototype extends RootModel {

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
   SimpleRlController rlController;

   HttpServer restServer;
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

      rlController = new SimpleRlController (
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

   private void startRestServer() throws IOException {
      if (restServer != null) {
         return;
      }

      restServer = HttpServer.create (new InetSocketAddress (restPort), 0);
      restServer.createContext ("/", new RootHandler());
      restServer.createContext ("/actionSize", new ActionSizeHandler());
      restServer.createContext ("/stateSize", new StateSizeHandler());
      restServer.createContext ("/obsSize", new ObsSizeHandler());
      restServer.createContext ("/excitations", new ExcitationsHandler());
      restServer.createContext ("/state", new StateHandler());
      restServer.createContext ("/time", new TimeHandler());
      restServer.createContext ("/reset", new ResetHandler());
      restServer.createContext ("/setSeed", new SetSeedHandler());
      restServer.createContext ("/setTest", new SetTestHandler());
      restServer.setExecutor (null);
      restServer.start();

      System.out.println (
         "MuscleExcitationPrototype REST server started at http://localhost:"
         + restPort);
   }

   private class RootHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            "{\"message\":\"Antagonist MuscleExcitationPrototype REST API\"}");
      }
   }

   private class ActionSizeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            Integer.toString (rlController.getActionSize()));
      }
   }

   private class StateSizeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            Integer.toString (rlController.getStateSize()));
      }
   }

   private class ObsSizeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            Integer.toString (rlController.getObservationSize()));
      }
   }

   private class StateHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (exchange, 200, rlController.getStateJson());
      }
   }

   private class TimeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            String.format (Locale.US, "%.6f", Main.getMain().getTime()));
      }
   }

   private class ResetHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         if (!exchange.getRequestMethod().equalsIgnoreCase ("POST")) {
            sendText (exchange, 405, "{\"error\":\"Use POST\"}");
            return;
         }

         resetEpisode();
         sendText (exchange, 200, rlController.getStateJson());
      }
   }

   private class ExcitationsHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         if (exchange.getRequestMethod().equalsIgnoreCase ("GET")) {
            sendText (exchange, 200, rlController.getExcitationsJson());
            return;
         }

         if (!exchange.getRequestMethod().equalsIgnoreCase ("POST")) {
            sendText (exchange, 405, "{\"error\":\"Use GET or POST\"}");
            return;
         }

         try {
            String body = readBody (exchange);
            double[] actions = parseNumbers (body);
            rlController.setExcitations (actions);
            sendText (exchange, 200, rlController.getStateJson());
         }
         catch (Exception e) {
            sendText (
               exchange, 400,
               "{\"error\":\"Could not parse excitation values\"}");
         }
      }
   }

   private class SetSeedHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         if (!exchange.getRequestMethod().equalsIgnoreCase ("POST")) {
            sendText (exchange, 405, "{\"error\":\"Use POST\"}");
            return;
         }

         try {
            String body = readBody (exchange);
            double[] values = parseNumbers (body);
            long seed = Math.round (values[0]);
            random.setSeed (seed);
            sendText (
               exchange, 200,
               String.format (Locale.US, "{\"seed\":%d}", seed));
         }
         catch (Exception e) {
            sendText (exchange, 400, "{\"error\":\"Could not parse seed\"}");
         }
      }
   }

   private class SetTestHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         if (!exchange.getRequestMethod().equalsIgnoreCase ("POST")) {
            sendText (exchange, 405, "{\"error\":\"Use POST\"}");
            return;
         }

         String body = readBody (exchange).trim().toLowerCase();
         testMode = body.indexOf ("true") != -1;
         sendText (
            exchange, 200,
            String.format (Locale.US, "{\"test\":%s}", testMode));
      }
   }

   private String readBody (HttpExchange exchange) throws IOException {
      BufferedReader reader = new BufferedReader (
         new InputStreamReader (
            exchange.getRequestBody(), StandardCharsets.UTF_8));

      StringBuilder sb = new StringBuilder();
      String line;
      while ((line = reader.readLine()) != null) {
         sb.append (line);
      }
      return sb.toString();
   }

   private double[] parseNumbers (String text) {
      Pattern pattern = Pattern.compile (
         "[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?");
      Matcher matcher = pattern.matcher (text);

      ArrayList<Double> values = new ArrayList<Double>();
      while (matcher.find()) {
         values.add (Double.parseDouble (matcher.group()));
      }

      if (values.size() == 0) {
         throw new IllegalArgumentException ("No number found");
      }

      double[] nums = new double[values.size()];
      for (int i = 0; i < values.size(); i++) {
         nums[i] = values.get (i);
      }
      return nums;
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

   private void sendText (
      HttpExchange exchange, int status, String response) throws IOException {

      byte[] bytes = response.getBytes (StandardCharsets.UTF_8);
      exchange.getResponseHeaders().set (
         "Content-Type", "application/json; charset=utf-8");
      exchange.getResponseHeaders().set ("Access-Control-Allow-Origin", "*");
      exchange.sendResponseHeaders (status, bytes.length);

      OutputStream os = exchange.getResponseBody();
      os.write (bytes);
      os.close();
   }
}
