"""Language Detection & Translation Utility.

Provides the Detect → Translate → Verify workflow for multilingual
fake news detection. Uses `langdetect` for language identification
and `deep-translator` (Google Translate) for translation.

Supported flow:
  1.  detect_language(text) → ISO 639-1 code (e.g. "hi", "fr", "zh-cn")
  2.  translate_to_english(text, source_lang) → English text
  3.  detect_and_translate(text) → LanguageResult with all metadata
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from .redaction import redact_secrets

logger = structlog.get_logger()

# ISO-639-1 code → human-readable name (covers the most common languages)
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}


@dataclass
class LanguageResult:
    """Result of language detection and optional translation."""
    original_text: str
    processed_text: str          # English text (translated or original)
    detected_language: str       # ISO 639-1 code
    language_name: str           # Human-readable name
    was_translated: bool         # True if the text was translated
    translation_failed: bool     # True if translation was attempted but failed


def detect_language(text: str) -> str:
    """Detect the language of `text`. Returns ISO 639-1 code.

    Falls back to 'en' on any error so the pipeline never breaks.
    """
    try:
        from langdetect import detect, LangDetectException
        code = detect(text)
        return code
    except Exception as exc:
        logger.warning("language.detect_error", error=redact_secrets(str(exc)))
        return "en"


# langdetect returns lowercase codes, but deep-translator (Google Translate)
# expects specific casing for some languages. This map handles the mismatch.
_LANG_CODE_REMAP: dict[str, str] = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "he": "iw",       # Google uses "iw" for Hebrew
    "jw": "jv",       # Javanese
}


def translate_to_english(text: str, source_lang: str) -> str | None:
    """Translate `text` from `source_lang` → English using Google Translate.

    Returns the translated string, or None on failure.
    """
    try:
        from deep_translator import GoogleTranslator

        # Remap language code if needed
        api_lang = _LANG_CODE_REMAP.get(source_lang, source_lang)

        # Google Translate has a ~5 000-char limit per request.
        # For longer texts we chunk and re-join.
        MAX_CHUNK = 4500
        if len(text) <= MAX_CHUNK:
            return GoogleTranslator(source=api_lang, target="en").translate(text)

        chunks: list[str] = []
        for i in range(0, len(text), MAX_CHUNK):
            chunk = text[i : i + MAX_CHUNK]
            translated_chunk = GoogleTranslator(source=api_lang, target="en").translate(chunk)
            chunks.append(translated_chunk or "")
        return " ".join(chunks)

    except Exception as exc:
        logger.warning(
            "language.translate_error",
            source=source_lang,
            error=redact_secrets(str(exc)),
        )
        # Fallback to LLM translation if available
        try:
            from ..config import settings
            if settings.gemini_api_key:
                from google import genai
                client = genai.Client(api_key=settings.gemini_api_key)
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=f"Translate the following text into clear English. Output ONLY the English translation, with no explanation, commentary, or markdown formatting:\n\n{text[:4000]}"
                )
                if response and response.text:
                    translated_text = response.text.strip()
                    logger.info("language.llm_translated", source=source_lang, length=len(translated_text))
                    return translated_text
        except Exception as llm_err:
            logger.warning(
                "language.llm_translate_failed",
                error=redact_secrets(str(llm_err)),
            )
        return None



def detect_and_translate(text: str) -> LanguageResult:
    """One-shot: detect the language, translate to English if needed.

    Always returns a valid LanguageResult — never raises.
    """
    detected = detect_language(text)
    lang_name = LANGUAGE_NAMES.get(detected, detected.upper())

    if detected == "en":
        return LanguageResult(
            original_text=text,
            processed_text=text,
            detected_language="en",
            language_name="English",
            was_translated=False,
            translation_failed=False,
        )

    logger.info("language.non_english_detected", lang=detected, name=lang_name)

    translated = translate_to_english(text, detected)

    if translated:
        logger.info("language.translated", lang=detected, original_len=len(text), translated_len=len(translated))
        return LanguageResult(
            original_text=text,
            processed_text=translated,
            detected_language=detected,
            language_name=lang_name,
            was_translated=True,
            translation_failed=False,
        )

    # Translation failed — fall back to the original text
    logger.warning("language.translation_fallback", lang=detected)
    return LanguageResult(
        original_text=text,
        processed_text=text,
        detected_language=detected,
        language_name=lang_name,
        was_translated=False,
        translation_failed=True,
    )
