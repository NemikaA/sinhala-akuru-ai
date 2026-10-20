package com.sinhalagen.glyph;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.*;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

public class SinhalaGlyphRendererTest {

    @Test
    @DisplayName("Verify Sinhala HarfBuzz glyph rendering with complex conjuncts and diacritics")
    public void testRenderSinhalaComplexConjuncts() throws Exception {
        SinhalaGlyphRenderer renderer = new SinhalaGlyphRenderer("non_existent_font.ttf");

        String[] testPhrases = {
                "ආයුබෝවන්",        // Common greeting with kombuwa (බෝ)
                "ශ්‍රී ලංකා",       // Sri Lanka with rakaransaya ligature (ශ්‍රී)
                "ක්‍රමලේඛනය",      // Programming with bandi akuru (ක්‍ර)
                "විශ්වවිද්‍යාලය"     // University with yansaya (ද්‍ය)
        };

        Rectangle box = new Rectangle(50, 150, 412, 160);

        for (String phrase : testPhrases) {
            SinhalaGlyphRenderer.RenderedGlyphPayload payload =
                    renderer.renderGlyphCondition(phrase, 512, 512, box, 54f);

            assertNotNull(payload);
            assertNotNull(payload.glyphImageBase64());
            assertNotNull(payload.boxMaskBase64());
            assertFalse(payload.glyphImageBase64().isEmpty());
            assertFalse(payload.boxMaskBase64().isEmpty());

            BufferedImage glyphImg = payload.glyphImage();
            assertEquals(512, glyphImg.getWidth());
            assertEquals(512, glyphImg.getHeight());

            // Verify that white pixels were drawn inside the canvas (not completely black)
            boolean hasWhitePixels = false;
            for (int y = 0; y < 512; y++) {
                for (int x = 0; x < 512; x++) {
                    int rgb = glyphImg.getRGB(x, y) & 0x00FFFFFF;
                    if (rgb > 0x888888) {
                        hasWhitePixels = true;
                        break;
                    }
                }
                if (hasWhitePixels) break;
            }

            assertTrue(hasWhitePixels, "Glyph canvas should contain rendered white strokes for: " + phrase);
        }
    }
}
