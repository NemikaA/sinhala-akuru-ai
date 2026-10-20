package com.sinhalagen;

import com.sinhalagen.client.InferenceClient;
import com.sinhalagen.glyph.SinhalaGlyphRenderer;

import javax.imageio.ImageIO;
import java.awt.Rectangle;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Main application entrypoint for Sinhala Akuru AI Java client.
 */
public class MainApp {

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  Sinhala Akuru AI (සිංහල අකුරු AI) Client");
        System.out.println("  Accurate Sinhala Visual Glyph Synthesis Pipeline");
        System.out.println("==================================================");

        String sampleText = "ආයුබෝවන්";
        String samplePrompt = "a vintage wooden signboard hanging outside a traditional Sri Lankan shop in Colombo, warm evening sunlight";
        String fontPath = "ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf";

        if (args.length > 0 && "--help".equalsIgnoreCase(args[0])) {
            printHelp();
            return;
        }

        boolean runInference = false;
        for (String arg : args) {
            if ("--infer".equalsIgnoreCase(arg)) {
                runInference = true;
            }
        }

        try {
            System.out.println("[1/2] Initializing Sinhala Glyph Renderer (HarfBuzz)...");
            SinhalaGlyphRenderer renderer = new SinhalaGlyphRenderer(fontPath);

            System.out.println("[2/2] Rendering sample glyph for: \"" + sampleText + "\"");
            Rectangle box = new Rectangle(56, 176, 400, 160);
            var payload = renderer.renderGlyphCondition(sampleText, 512, 512, box, 64f);

            File outDir = new File("output");
            outDir.mkdirs();

            File glyphFile = new File(outDir, "test_glyph_mask.png");
            File maskFile = new File(outDir, "test_region_box.png");
            ImageIO.write(payload.glyphImage(), "png", glyphFile);
            ImageIO.write(payload.boxMask(), "png", maskFile);

            System.out.println("[OK] Glyph condition saved to: " + glyphFile.getAbsolutePath());
            System.out.println("[OK] Region bounding mask saved to: " + maskFile.getAbsolutePath());

            if (runInference) {
                System.out.println("\nConnecting to Python Inference Server (http://127.0.0.1:8000)...");
                InferenceClient client = new InferenceClient(fontPath);
                Path outputImg = Paths.get("output", "generated_sinhala_scene.png");
                client.generateSinhalaScene(samplePrompt, sampleText, outputImg);
                System.out.println("[OK] Scene image generated: " + outputImg.toAbsolutePath());
            } else {
                System.out.println("\nTip: To run full diffusion inference with the Python engine:");
                System.out.println("  1. Start the ML engine: ./gradlew startInferenceServer");
                System.out.println("  2. Run inference: ./gradlew run --args=\"--infer\"");
            }

        } catch (Exception e) {
            System.err.println("Error during execution: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void printHelp() {
        System.out.println("Usage:");
        System.out.println("  ./gradlew run                  # Generates glyph & mask preview PNGs");
        System.out.println("  ./gradlew run --args=\"--infer\"  # Calls FastAPI inference server to generate scene");
    }
}
