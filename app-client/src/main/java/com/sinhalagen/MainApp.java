package com.sinhalagen;

import com.sinhalagen.client.InferenceClient;
import com.sinhalagen.glyph.SinhalaGlyphRenderer;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Main application entrypoint for Sinhala Akuru AI.
 * Demonstrates inserting stylized Sinhala text onto ANY image while leaving
 * the background image 100% intact.
 */
public class MainApp {

    public static void main(String[] args) {
        System.out.println("==========================================================");
        System.out.println("  Sinhala Akuru AI - Visual Text Inpainting & Styling");
        System.out.println("  Paints stylized Sinhala text onto ANY intact image");
        System.out.println("==========================================================");

        String sampleText = "\u0D86\u0DBA\u0DD4\u0DB6\u0DDD\u0DC0\u0DB1\u0DCA \u0DC1\u0DCA\u200D\u0DBB\u0DD3 \u0DBD\u0D82\u0D9A\u0DCF";
        String fontPath = "ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf";
        String chosenStyle = "neon_glow";
        String inputImagePath = null;
        boolean runInference = false;

        for (int i = 0; i < args.length; i++) {
            if ("--text".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                sampleText = args[++i];
            } else if ("--style".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                chosenStyle = args[++i];
            } else if ("--image".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                inputImagePath = args[++i];
                runInference = true;
            } else if ("--infer".equalsIgnoreCase(args[i])) {
                runInference = true;
            }
        }

        try {
            File outDir = new File("output");
            outDir.mkdirs();

            // 1. If no image was provided, create a sample background photo (e.g. wall/signboard)
            File baseImageFile;
            if (inputImagePath != null) {
                baseImageFile = new File(inputImagePath);
            } else {
                baseImageFile = new File(outDir, "sample_storefront_wall.png");
                createSamplePhoto(baseImageFile);
                System.out.println("[INFO] Created sample background image at: " + baseImageFile.getAbsolutePath());
            }

            // 2. Generate crisp glyph mask via HarfBuzz
            System.out.println("[1/2] Generating accurate Sinhala glyph mask for: \"" + sampleText + "\"");
            SinhalaGlyphRenderer renderer = new SinhalaGlyphRenderer(fontPath);
            Rectangle box = new Rectangle(20, 160, 472, 160);
            var payload = renderer.renderGlyphCondition(sampleText, 512, 512, box, 44f);

            File glyphFile = new File(outDir, "sinhala_glyph_mask.png");
            ImageIO.write(payload.glyphImage(), "png", glyphFile);
            System.out.println("[OK] Glyph mask ready: " + glyphFile.getAbsolutePath());

            // 3. Connect to inference server to apply stylized text while keeping base image intact
            if (runInference) {
                System.out.println("[2/2] Inpainting stylized Sinhala text ('" + chosenStyle + "') onto base photo...");
                InferenceClient client = new InferenceClient(fontPath);
                Path outputImg = Paths.get("output", "intact_result_" + chosenStyle + ".png");
                client.insertSinhalaTextIntoImage(baseImageFile.toPath(), sampleText, null, chosenStyle, null, outputImg);
                System.out.println("\n[SUCCESS] Intact image with stylized Sinhala text saved to:");
                System.out.println("          " + outputImg.toAbsolutePath());
            } else {
                System.out.println("\nTip: Start the server and run full intact image inpainting with:");
                System.out.println("  1. In terminal 1: py -m uvicorn server.inference_server:app --port 8000 (inside ml-engine)");
                System.out.println("  2. In terminal 2: ./gradlew run --args=\"--infer --style neon_glow\"");
                System.out.println("  (Styles: neon_glow, wood_carved, metallic_gold, painted_sign, chalk)");
            }

        } catch (Exception e) {
            System.err.println("Error during execution: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void createSamplePhoto(File targetFile) throws Exception {
        BufferedImage img = new BufferedImage(700, 450, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();

        // Draw wooden / architectural textures
        for (int y = 0; y < 450; y += 4) {
            int tone = 120 + (int) (Math.random() * 25);
            g.setColor(new Color(tone, (int) (tone * 0.72), (int) (tone * 0.45)));
            g.fillRect(0, y, 700, 4);
        }

        // Draw decorative borders of a shop signboard
        g.setColor(new Color(60, 40, 20));
        g.setStroke(new BasicStroke(6));
        g.drawRect(20, 20, 660, 410);

        g.dispose();
        ImageIO.write(img, "png", targetFile);
    }
}
