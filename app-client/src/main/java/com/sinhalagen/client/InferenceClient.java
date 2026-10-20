package com.sinhalagen.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sinhalagen.glyph.SinhalaGlyphRenderer;

import java.awt.Rectangle;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

/**
 * Java client for communicating with the Python FastAPI Sinhala Glyph Inference Server.
 * Supports inserting stylized Sinhala text onto ANY user-provided image while keeping
 * the base image completely intact.
 */
public class InferenceClient {

    private final HttpClient httpClient;
    private final ObjectMapper mapper;
    private final SinhalaGlyphRenderer renderer;
    private final String serverUrl;

    public InferenceClient(String fontPath) throws Exception {
        this(fontPath, "http://127.0.0.1:8000");
    }

    public InferenceClient(String fontPath, String serverUrl) throws Exception {
        this.renderer = new SinhalaGlyphRenderer(fontPath);
        this.serverUrl = serverUrl;
        this.mapper = new ObjectMapper();
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    /**
     * Inserts accurate, stylized Sinhala text onto an existing image while keeping
     * every other pixel of the image completely intact.
     *
     * @param inputImagePath  Path to the user's base photo
     * @param sinhalaText     Sinhala text to render (e.g. "ආයුබෝවන්", "සිංහල තේ කඩේ")
     * @param targetBox       Optional bounding box placement (can be null for auto-center)
     * @param style           Style preset (e.g. "wood_carved", "neon_glow", "metallic_gold", "painted_sign", "chalk")
     * @param colorHex        Optional custom hex color (e.g. "#00FFFF", or null)
     * @param outputPath      Path to save the modified image
     */
    public void insertSinhalaTextIntoImage(
            Path inputImagePath,
            String sinhalaText,
            Rectangle targetBox,
            String style,
            String colorHex,
            Path outputPath) throws Exception {

        if (!Files.exists(inputImagePath)) {
            throw new IOException("Input image does not exist: " + inputImagePath.toAbsolutePath());
        }

        // 1. Read input image and encode to Base64
        byte[] inputImageBytes = Files.readAllBytes(inputImagePath);
        String baseImageBase64 = Base64.getEncoder().encodeToString(inputImageBytes);

        // 2. Render deterministic Sinhala glyph layout via HarfBuzz / Uniscribe
        Rectangle renderBox = targetBox != null ? targetBox : new Rectangle(20, 160, 472, 160);
        var payload = renderer.renderGlyphCondition(sinhalaText, 512, 512, renderBox, 48f);

        // 3. Build JSON payload
        Map<String, Object> reqMap = new HashMap<>();
        reqMap.put("glyph_mask_base64", payload.glyphImageBase64());
        reqMap.put("base_image_base64", baseImageBase64);
        reqMap.put("style", style != null ? style : "wood_carved");
        if (colorHex != null) {
            reqMap.put("color_hex", colorHex);
        }
        if (targetBox != null) {
            reqMap.put("box_x", targetBox.x);
            reqMap.put("box_y", targetBox.y);
            reqMap.put("box_w", targetBox.width);
            reqMap.put("box_h", targetBox.height);
        }

        String requestBody = mapper.writeValueAsString(reqMap);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/generate"))
                .timeout(Duration.ofMinutes(1))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Server error (" + response.statusCode() + "): " + response.body());
        }

        JsonNode root = mapper.readTree(response.body());
        byte[] imageBytes = Base64.getDecoder().decode(root.get("generated_image_base64").asText());

        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        Files.write(outputPath, imageBytes);
        System.out.println("[OK] Successfully generated intact image with stylized Sinhala text:");
        System.out.println("     " + outputPath.toAbsolutePath());
    }

    /**
     * Standalone scene generation without base image.
     */
    public void generateSinhalaScene(String scenePrompt, String sinhalaText, Path outputPath) throws Exception {
        Rectangle defaultBox = new Rectangle(50, 180, 412, 140);
        var payload = renderer.renderGlyphCondition(sinhalaText, 512, 512, defaultBox, 48f);

        String requestBody = mapper.writeValueAsString(Map.of(
                "scene_prompt", scenePrompt,
                "glyph_mask_base64", payload.glyphImageBase64(),
                "style", "metallic_gold",
                "controlnet_scale", 1.15,
                "steps", 25
        ));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/generate"))
                .timeout(Duration.ofMinutes(2))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new IOException("Server error (" + response.statusCode() + "): " + response.body());
        }

        JsonNode root = mapper.readTree(response.body());
        byte[] imageBytes = Base64.getDecoder().decode(root.get("generated_image_base64").asText());
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        Files.write(outputPath, imageBytes);
        System.out.println("[OK] Scene generated to: " + outputPath.toAbsolutePath());
    }

    public SinhalaGlyphRenderer getRenderer() {
        return renderer;
    }
}
