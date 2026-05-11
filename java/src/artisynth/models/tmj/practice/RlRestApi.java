package artisynth.models.tmj.practice;

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

import artisynth.core.driver.Main;

/**
 * REST layer used by the Python RL environment.
 * Model-specific reset/seed/test behavior is supplied through ModelHooks.
 */
public class RlRestApi {

   /**
    * Hooks implemented by the concrete ArtiSynth model.
    */
   public static interface ModelHooks {
      public void resetEpisode();

      public void setSeed (long seed);

      public void setTestMode (boolean testMode);
   }

   private int port;
   private RlController controller;
   private ModelHooks hooks;
   private HttpServer server;

   public RlRestApi (
      int port, RlController controller, ModelHooks hooks) {

      if (controller == null) {
         throw new IllegalArgumentException ("Controller cannot be null");
      }
      if (hooks == null) {
         throw new IllegalArgumentException ("Model hooks cannot be null");
      }

      this.port = port;
      this.controller = controller;
      this.hooks = hooks;
   }

   public synchronized void start() throws IOException {
      if (server != null) {
         return;
      }

      server = HttpServer.create (new InetSocketAddress (port), 0);
      server.createContext ("/", new RootHandler());
      server.createContext ("/actionSize", new ActionSizeHandler());
      server.createContext ("/stateSize", new StateSizeHandler());
      server.createContext ("/obsSize", new ObsSizeHandler());
      server.createContext ("/excitations", new ExcitationsHandler());
      server.createContext ("/state", new StateHandler());
      server.createContext ("/time", new TimeHandler());
      server.createContext ("/reset", new ResetHandler());
      server.createContext ("/setSeed", new SetSeedHandler());
      server.createContext ("/setTest", new SetTestHandler());
      server.setExecutor (null);
      server.start();

      System.out.println (
         "ArtiSynth RL REST API started at http://localhost:" + port);
   }

   public synchronized void stop() {
      if (server != null) {
         server.stop (0);
         server = null;
      }
   }

   private class RootHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (exchange, 200, "{\"message\":\"ArtiSynth RL REST API\"}");
      }
   }

   private class ActionSizeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            Integer.toString (controller.getActionSize()));
      }
   }

   private class StateSizeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            Integer.toString (controller.getStateSize()));
      }
   }

   private class ObsSizeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            Integer.toString (controller.getObservationSize()));
      }
   }

   private class StateHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (exchange, 200, controller.getStateJson());
      }
   }

   private class TimeHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         sendText (
            exchange, 200,
            String.format (Locale.US, "%.6f", getCurrentTime()));
      }
   }

   private class ResetHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         if (!exchange.getRequestMethod().equalsIgnoreCase ("POST")) {
            sendText (exchange, 405, "{\"error\":\"Use POST\"}");
            return;
         }

         hooks.resetEpisode();
         sendText (exchange, 200, controller.getStateJson());
      }
   }

   private class ExcitationsHandler implements HttpHandler {
      public void handle (HttpExchange exchange) throws IOException {
         if (exchange.getRequestMethod().equalsIgnoreCase ("GET")) {
            sendText (exchange, 200, controller.getExcitationsJson());
            return;
         }

         if (!exchange.getRequestMethod().equalsIgnoreCase ("POST")) {
            sendText (exchange, 405, "{\"error\":\"Use GET or POST\"}");
            return;
         }

         try {
            String body = readBody (exchange);
            double[] actions = parseNumbers (body);
            controller.setExcitations (actions);
            sendText (exchange, 200, controller.getStateJson());
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
            hooks.setSeed (seed);
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
         boolean testMode = body.indexOf ("true") != -1;
         hooks.setTestMode (testMode);
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

   private double getCurrentTime() {
      Main main = Main.getMain();
      return main == null ? 0.0 : main.getTime();
   }
}
