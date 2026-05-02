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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import artisynth.core.gui.ControlPanel;
import artisynth.core.driver.Main;
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
   Particle anchor;
   Particle target;
   RigidBody box;
   FrameMarker marker;
   Muscle muscle;
   MuscleExciter exciter;
   SimpleRlController rlController;

   HttpServer restServer;
   int restPort = 8081;

   public void build (String[] args) throws IOException {

      mech = new MechModel ("mech");
      addModel (mech);

      mech.setGravity (0, 0, 0);

      anchor = new Particle ("anchor", 2, 0, 0, 0);
      anchor.setDynamic (false);

      box = RigidBody.createBox ("box", 0.5, 0.3, 0.3, 20);
      box.setPose (new RigidTransform3d (0.75, 0, 0));

      marker = new FrameMarker (-0.25, 0, 0);
      marker.setName ("marker");
      marker.setFrame (box);

      target = new Particle ("target", 1, 0.20, 0, 0);
      target.setDynamic (false);

      muscle = new Muscle ("muscle", 0);
      muscle.setPoints (anchor, marker);
      muscle.setMaterial (new SimpleAxialMuscle (20, 10, 10));

      mech.addParticle (anchor);
      mech.addParticle (target);
      mech.addRigidBody (box);
      mech.addFrameMarker (marker);
      mech.addAxialSpring (muscle);

      exciter = new MuscleExciter ("muscleExciter");
      exciter.addTarget (muscle, 1.0);
      mech.addMuscleExciter (exciter);

      rlController = new SimpleRlController (marker, target, muscle, exciter);
      addController (rlController);

      addControlPanel();
      addOutputProbes();
      startRestServer();

      addTracingProbe (marker, "position", 0, 30);

      mech.setBounds (-1, 0, -1, 1, 0, 1);

      RenderProps.setSphericalPoints (anchor, 0.06, Color.BLUE);
      RenderProps.setSphericalPoints (marker, 0.06, Color.BLUE);
      RenderProps.setSphericalPoints (target, 0.07, Color.GREEN);
      RenderProps.setSpindleLines (muscle, 0.02, Color.RED);
      RenderProps.setFaceColor (box, Color.LIGHT_GRAY);
   }

   private void addControlPanel() {
      ControlPanel panel = new ControlPanel ("controls");
      panel.addWidget (
         "external action", rlController, "actionExcitation", 0.0, 1.0);
      panel.addWidget ("exciter excitation", exciter, "excitation");
      panel.addWidget ("muscle excitation", muscle, "excitation");
      panel.addWidget (mech, "gravity");
      panel.addWidget (box, "mass");
      addControlPanel (panel);
   }

   private void addOutputProbes() {
      addProbe ("marker position", marker, "position");
      addProbe ("marker velocity", marker, "velocity");
      addProbe ("target position", target, "position");
      addProbe ("exciter excitation", exciter, "excitation");
      addProbe ("muscle excitation", muscle, "excitation");
      addProbe ("muscle force", muscle, "forceNorm");
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
            "{\"message\":\"MuscleExcitationPrototype REST API\"}");
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
      rlController.setExcitations (new double[] { 0.0 });
      exciter.setExcitation (0.0);
      muscle.setExcitation (0.0);

      box.setPose (new RigidTransform3d (0.75, 0, 0));
      box.setVelocity (0, 0, 0, 0, 0, 0);

      target.setPosition (0.20, 0, 0);
      target.setVelocity (0, 0, 0);
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
