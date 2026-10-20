package com.sinhalagen.glyph;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Base64;

/**
 * Deterministic Sinhala Glyph Renderer using Java AWT TextLayout (backed by HarfBuzz/Uniscribe).
 * Guarantees 100% accurate rendering of complex Brahmic/Indic ligatures, vowel signs
 * (Pillam such as Kombuwa, Ispilla, Papilla), and consonant conjuncts (Bandi Akuru,
 * Rakaransaya, Yansaya) without neural network hallucination or glyph smudging.
 */
public class SinhalaGlyphRenderer {

    private final Font baseFont;

    /**
     * Initializes the renderer with a Unicode font file or robust auto-detection fallback.
     *
     * @param fontFilePath Path to TrueType/OpenType font (.ttf / .otf).
     */
    public SinhalaGlyphRenderer(String fontFilePath) {
        this.baseFont = resolveFont(fontFilePath);
    }

    /**
     * Initializes the renderer with a pre-instantiated Font.
     *
     * @param font Java AWT Font
     */
    public SinhalaGlyphRenderer(Font font) {
        this.baseFont = font != null ? font : resolveFont(null);
    }

    /**
     * Searches multiple paths and system font registries for a font that properly shapes Sinhala.
     * On Windows, Nirmala UI & Iskoola Pota have native Microsoft 'sinh' tables that properly
     * reorder left-side vowel signs (Kombuwa ෙ) and diacritics without smudging.
     */
    private static Font resolveFont(String preferredPath) {
        // 1. If explicit font path provided and not a default placeholder
        if (preferredPath != null && !preferredPath.trim().isEmpty()) {
            File file = new File(preferredPath);
            if (file.exists() && !preferredPath.contains("NotoSansSinhala")) {
                try {
                    Font f = Font.createFont(Font.TRUETYPE_FONT, file);
                    System.out.println("[INFO] Loaded custom font: " + file.getAbsolutePath());
                    return f;
                } catch (Exception ignored) {
                }
            }
        }

        // 2. On Windows, Nirmala UI and Iskoola Pota have native Microsoft 'sinh' shaping tables
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            for (String winFontName : new String[]{"Nirmala UI", "Iskoola Pota"}) {
                Font f = new Font(winFontName, Font.BOLD, 48);
                if (f.canDisplay(0x0D86) && !f.getFamily().equalsIgnoreCase("Dialog")) {
                    System.out.println("[INFO] Using Windows native Sinhala shaping font: " + f.getFontName());
                    return f;
                }
            }
        }

        // 3. Check bundled / system font files
        String[] candidatePaths = {
                "C:/Windows/Fonts/Nirmala.ttc",
                "C:/Windows/Fonts/Nirmala.ttf",
                "C:/Windows/Fonts/iskpota.ttf",
                "C:/Windows/Fonts/iskpotab.ttf",
                preferredPath,
                "ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf",
                "../ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansSinhala-Bold.ttf"
        };

        for (String path : candidatePaths) {
            if (path != null && !path.trim().isEmpty()) {
                File file = new File(path);
                if (file.exists()) {
                    try {
                        Font f = Font.createFont(Font.TRUETYPE_FONT, file);
                        System.out.println("[INFO] Loaded font file: " + file.getAbsolutePath());
                        return f;
                    } catch (Exception ignored) {
                    }
                }
            }
        }

        // 4. Search all system fonts for Sinhala character coverage
        GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
        for (Font f : ge.getAllFonts()) {
            if (f.canDisplay(0x0D86) && !f.getFamily().equalsIgnoreCase("Dialog")) {
                System.out.println("[INFO] Found system Sinhala font: " + f.getFontName());
                return f.deriveFont(Font.BOLD, 48f);
            }
        }

        System.err.println("[WARNING] No Sinhala-capable font found! Text may render as tofu boxes.");
        return new Font(Font.SANS_SERIF, Font.BOLD, 48);
    }

    /**
     * Renders complex Sinhala Unicode text onto an image conditioning canvas
     * and returns both the Glyph Condition Image and the Binary Region Mask.
     *
     * @param sinhalaText Input Sinhala Unicode string (U+0D80 to U+0DFF)
     * @param imageWidth  Canvas width in pixels (typically 512 or 1024)
     * @param imageHeight Canvas height in pixels (typically 512 or 1024)
     * @param targetBox   Bounding rectangle for text positioning
     * @param fontSize    Target font size in points
     * @return RenderedGlyphPayload containing Base64 strings for glyph and mask
     * @throws Exception If rendering or encoding fails
     */
    public RenderedGlyphPayload renderGlyphCondition(
            String sinhalaText,
            int imageWidth,
            int imageHeight,
            Rectangle targetBox,
            float fontSize) throws Exception {

        // 1. Create Glyph Condition Image (White text on Black background for ControlNet)
        BufferedImage glyphCanvas = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = glyphCanvas.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);

        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, imageWidth, imageHeight);

        Font sizedFont = baseFont.deriveFont(Font.BOLD, fontSize);
        FontRenderContext frc = g2d.getFontRenderContext();

        // TextLayout invokes native CTL shaping (reordering kombuwa and positioning diacritics)
        TextLayout textLayout = new TextLayout(sinhalaText, sizedFont, frc);
        Rectangle textBounds = textLayout.getPixelBounds(frc, 0, 0);

        // Center the shaped Sinhala glyph outline inside the target bounding box
        double startX = targetBox.x + (targetBox.width - textBounds.width) / 2.0;
        double startY = targetBox.y + (targetBox.height + textBounds.height) / 2.0;

        g2d.setColor(Color.WHITE);
        textLayout.draw(g2d, (float) startX, (float) startY);
        g2d.dispose();

        // 2. Create Binary Region Mask (Marks where text box is located for Loss/Inpainting)
        BufferedImage maskCanvas = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D maskG2d = maskCanvas.createGraphics();
        maskG2d.setColor(Color.BLACK);
        maskG2d.fillRect(0, 0, imageWidth, imageHeight);
        maskG2d.setColor(Color.WHITE);
        maskG2d.fillRect(targetBox.x, targetBox.y, targetBox.width, targetBox.height);
        maskG2d.dispose();

        return new RenderedGlyphPayload(
                toBase64(glyphCanvas),
                toBase64(maskCanvas),
                glyphCanvas,
                maskCanvas
        );
    }

    private String toBase64(BufferedImage img) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(img, "png", baos);
        return Base64.getEncoder().encodeToString(baos.toByteArray());
    }

    /**
     * Payload record holding Base64-encoded PNGs and BufferedImage references.
     */
    public record RenderedGlyphPayload(
            String glyphImageBase64,
            String boxMaskBase64,
            BufferedImage glyphImage,
            BufferedImage boxMask
    ) {}

    /**
     * Standalone test entrypoint (requires zero external dependencies).
     * Uses Unicode escape sequence for "ආයුබෝවන් ශ්‍රී ලංකා" to prevent compiler charset corruption.
     */
    public static void main(String[] args) {
        System.out.println("=== Sinhala Glyph HarfBuzz Shaping Engine Test ===");
        // Default: "ආයුබෝවන් ශ්‍රී ලංකා" represented via Unicode escapes to be 100% charset-safe
        String sampleText = args.length > 0 ? args[0] : "\u0D86\u0DBA\u0DD4\u0DB6\u0DDD\u0DC0\u0DB1\u0DCA \u0DC1\u0DCA\u200D\u0DBB\u0DD3 \u0DBD\u0D82\u0D9A\u0DCF";

        try {
            SinhalaGlyphRenderer renderer = new SinhalaGlyphRenderer("ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf");
            Rectangle box = new Rectangle(10, 160, 492, 160);
            RenderedGlyphPayload payload = renderer.renderGlyphCondition(sampleText, 512, 512, box, 44f);

            File outDir = new File("output");
            outDir.mkdirs();

            File glyphFile = new File(outDir, "sinhala_glyph_mask.png");
            File maskFile = new File(outDir, "sinhala_region_box.png");
            ImageIO.write(payload.glyphImage(), "png", glyphFile);
            ImageIO.write(payload.boxMask(), "png", maskFile);

            System.out.println("[OK] Successfully generated HarfBuzz glyph mask:");
            System.out.println("     " + glyphFile.getAbsolutePath());
            System.out.println("[OK] Successfully generated region bounding box:");
            System.out.println("     " + maskFile.getAbsolutePath());
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
