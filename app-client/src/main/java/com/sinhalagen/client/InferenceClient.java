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
import java.util.Map;

/**
 * Java client for communicating with the Python FastAPI Sinhala Glyph Inference Server.
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
     * Renders Sinhala text into a glyph mask and sends it along with the prompt to the ML server.
     *
     * @param scenePrompt  English background/scene description (e.g., "A glowing neon sign in Colombo night rain")
     * @param sinhalaText  Sinhala Unicode string to render
     * @param outputPath   Output path for the generated image PNG
     * @throws Exception If rendering or network inference fails
     */
    public void generateSinhalaScene(String scenePrompt, String sinhalaText, Path outputPath) throws Exception {
        Rectangle defaultBox = new Rectangle(60, 180, 392, 140);
        generateSinhalaScene(scenePrompt, sinhalaText, defaultBox, 56f, 1.15, 25, outputPath);
    }

    /**
     * Full configuration generation call.
     */
    public void generateSinhalaScene(
            String scenePrompt,
            String sinhalaText,
            Rectangle textBox,
            float fontSize,
            double controlnetScale,
            int steps,
            Path outputPath) throws Exception {

        // 1. Render exact Sinhala glyph layout via HarfBuzz-backed TextLayout
        var payload = renderer.renderGlyphCondition(sinhalaText, 512, 512, textBox, fontSize);

        // 2. Build JSON payload for FastAPI server
        String requestBody = mapper.writeValueAsString(Map.of(
                "scene_prompt", scenePrompt,
                "glyph_mask_base64", payload.glyphImageBase64(),
                "controlnet_scale", controlnetScale,
                "steps", steps
        ));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/generate"))
                .timeout(Duration.ofMinutes(2))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Inference server error (" + response.statusCode() + "): " + response.body());
        }

        JsonNode root = mapper.readTree(response.body());
        byte[] imageBytes = Base64.getDecoder().decode(root.get("generated_image_base64").asText());

        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        Files.write(outputPath, imageBytes);
        System.out.println("Saved generated Sinhala scene image to: " + outputPath.toAbsolutePath());
    }

    public SinhalaGlyphRenderer getRenderer() {
        return renderer;
    }
}
