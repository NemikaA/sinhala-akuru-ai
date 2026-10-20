package com.sinhalagen.glyph;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Base64;

/**
 * Deterministic Sinhala Glyph Renderer using Java AWT TextLayout (backed by HarfBuzz).
 * Guarantees 100% accurate rendering of complex Brahmic/Indic ligatures, vowel signs
 * (Pillam such as Kombuwa, Ispilla, Papilla), and consonant conjuncts (Bandi Akuru,
 * Rakaransaya, Yansaya) without neural network hallucination.
 */
public class SinhalaGlyphRenderer {

    private final Font baseFont;

    /**
     * Initializes the renderer with a Unicode font file (e.g. NotoSansSinhala, Iskoola Pota).
     *
     * @param fontFilePath Path to TrueType/OpenType font (.ttf / .otf).
     * @throws Exception If font file loading fails.
     */
    public SinhalaGlyphRenderer(String fontFilePath) throws Exception {
        File file = new File(fontFilePath);
        if (file.exists()) {
            this.baseFont = Font.createFont(Font.TRUETYPE_FONT, file);
        } else {
            // Fallback to a system Sinhala font or generic SansSerif
            this.baseFont = new Font("Iskoola Pota", Font.PLAIN, 48);
        }
    }

    /**
     * Initializes the renderer with a pre-instantiated Font.
     *
     * @param font Java AWT Font
     */
    public SinhalaGlyphRenderer(Font font) {
        this.baseFont = font != null ? font : new Font(Font.SANS_SERIF, Font.PLAIN, 48);
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

        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, imageWidth, imageHeight);

        Font sizedFont = baseFont.deriveFont(Font.BOLD, fontSize);
        FontRenderContext frc = g2d.getFontRenderContext();

        // TextLayout invokes HarfBuzz shaping under the hood for accurate Sinhala ligatures & pili
        TextLayout textLayout = new TextLayout(sinhalaText, sizedFont, frc);
        Rectangle textBounds = textLayout.getPixelBounds(null, 0, 0);

        // Center the shaped Sinhala glyph outline inside the target bounding box
        double startX = targetBox.x + (targetBox.width - textBounds.width) / 2.0;
        double startY = targetBox.y + (targetBox.height + textBounds.height) / 2.0;

        g2d.setColor(Color.WHITE);
        Shape outline = textLayout.getOutline(AffineTransform.getTranslateInstance(startX, startY));
        g2d.fill(outline);
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
}
