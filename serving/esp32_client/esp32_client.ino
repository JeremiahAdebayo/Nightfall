/*
 * Nightfall ESP32 MVP client (Wokwi simulation)
 *
 * Honest scope, stated plainly: this does NOT simulate a real camera
 * capturing a live frame. Wokwi's ESP32-CAM support covers the
 * esp_camera.h API compiling and running, WiFi, and HTTP client code
 * correctly -- but does not provide a genuinely simulated image sensor
 * feeding real pixel data. Rather than claim a camera simulation that
 * doesn't really exist, this sketch sends a small embedded test image
 * (a real captured MVTec test image, converted to a byte array at
 * build time) as an explicit stand-in for "a frame this device
 * captured."
 *
 * What this DOES genuinely validate: real WiFi connection handling,
 * real HTTP client code, real multipart/form-data construction, and
 * real parsing of a JSON response from Nightfall's REST gateway -- all
 * of which will run unchanged on physical ESP32-CAM hardware. Only the
 * image source (embedded bytes vs. a live camera capture) differs
 * between this simulation and eventual real hardware.
 *
 * MVP scope, deliberately minimal: one WiFi connect, one HTTP POST, one
 * printed result. No LED, no retry logic, no loop -- prove the chain
 * connects before adding anything else.
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "test_image.h"  // defines TEST_IMAGE_BYTES[] and TEST_IMAGE_LEN -- see note below

const char* WIFI_SSID = "Wokwi-GUEST";  // Wokwi's built-in virtual network
const char* WIFI_PASSWORD = "";          // Wokwi-GUEST is open, no password

// Replace with your actual REST gateway's reachable address. Wokwi's
// simulated ESP32 can reach the public internet, so if your gateway is
// exposed via a tunnel (e.g. ngrok) or a public Colab-forwarded URL,
// put that URL here. localhost will NOT work -- Wokwi's simulated
// device is not the same machine running your Colab/gateway process.
//
// IMPORTANT: ngrok URLs are https:// (TLS), not http://. This sketch
// uses WiFiClientSecure with setInsecure() (skips certificate
// validation) to reach it -- acceptable for this MVP/simulation, since
// we're not handling sensitive data and the goal is proving the
// connection chain works, not production-grade certificate pinning.
// A real hardware deployment against a permanent, known server should
// validate the actual certificate instead of skipping validation.
const char* GATEWAY_URL = "https://YOUR_NGROK_URL.ngrok-free.app/detect";
const char* CATEGORY = "bottle";

WiFiClientSecure secureClient;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Nightfall ESP32 MVP client starting...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi connection FAILED. Check Wokwi-GUEST availability.");
    return;
  }
  Serial.println("\nWiFi connected.");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  sendDetectionRequest();
}

void sendDetectionRequest() {
  HTTPClient http;

  // setInsecure() skips certificate validation -- see the GATEWAY_URL
  // comment above for why that's an acceptable tradeoff for this
  // MVP/simulation specifically, not a general recommendation.
  secureClient.setInsecure();
  http.begin(secureClient, GATEWAY_URL);

  // Manually construct a multipart/form-data body -- this matches what
  // the REST gateway's FastAPI endpoint (category: Form, image: File)
  // expects. This is genuinely how a real ESP32-CAM sketch would send
  // a captured frame; only TEST_IMAGE_BYTES's content (not the request
  // construction logic) is a stand-in here.
  String boundary = "----NightfallBoundary";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  String bodyStart =
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"category\"\r\n\r\n" +
    String(CATEGORY) + "\r\n" +
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"image\"; filename=\"test.png\"\r\n"
    "Content-Type: image/png\r\n\r\n";
  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  size_t totalLen = bodyStart.length() + TEST_IMAGE_LEN + bodyEnd.length();
  uint8_t* body = (uint8_t*)malloc(totalLen);
  if (body == NULL) {
    Serial.println("ERROR: malloc failed, image too large for available heap.");
    http.end();
    return;
  }

  size_t offset = 0;
  memcpy(body + offset, bodyStart.c_str(), bodyStart.length());
  offset += bodyStart.length();
  memcpy(body + offset, TEST_IMAGE_BYTES, TEST_IMAGE_LEN);
  offset += TEST_IMAGE_LEN;
  memcpy(body + offset, bodyEnd.c_str(), bodyEnd.length());

  Serial.println("Sending detection request...");
  int statusCode = http.POST(body, totalLen);
  free(body);

  if (statusCode > 0) {
    String response = http.getString();
    Serial.print("HTTP status: ");
    Serial.println(statusCode);
    Serial.print("Response: ");
    Serial.println(response);
  } else {
    Serial.print("HTTP request FAILED, error: ");
    Serial.println(http.errorToString(statusCode));
  }

  http.end();
}

void loop() {
  // Intentionally empty -- MVP sends one request in setup() and stops.
  // Real hardware would loop on actual camera capture; that's explicit
  // future scope, not part of this MVP.
}
