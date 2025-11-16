"""
Internationalization (i18n) module for Air Draw Classifier.

Provides multi-language support using gettext for UI text translation.
Soporta español, inglés y puede extenderse fácilmente a otros idiomas.
"""

import gettext
import locale
import os
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class I18nManager:
    """
    Manager for internationalization support.

    Handles language detection, translation loading, and text translation.
    Soporta carga dinámica de idiomas y fallback automático.
    """

    def __init__(self, locale_dir: str = "locale"):
        """
        Initialize the i18n manager.

        Args:
            locale_dir: Directory containing translation files
        """
        self.locale_dir = Path(locale_dir)
        self.current_translator = None
        self.current_language = None
        self.fallback_language = 'en'
        
        # Diccionario de traducciones en memoria para uso offline
        self.translation_cache: Dict[str, Dict[str, str]] = {}
        
        # Idiomas soportados
        self.supported_languages = {
            'es': 'Español',
            'en': 'English',
            'fr': 'Français',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português'
        }

        # Detect system language
        self.system_language = self._detect_system_language()

        # Load default language (English)
        self.load_language("en")

    def _detect_system_language(self) -> str:
        """
        Detect the system language.

        Returns:
            Language code (e.g., 'es', 'en')
        """
        try:
            # Get system locale
            system_locale, _ = locale.getlocale()

            if system_locale:
                # Extract language code (first part before underscore)
                language = system_locale.split('_')[0].lower()

                # Map common languages
                if language in ['es', 'spanish']:
                    return 'es'
                elif language in ['en', 'english']:
                    return 'en'
                else:
                    # Default to English for unsupported languages
                    return 'en'
            else:
                return 'en'
        except:
            # Fallback to English
            return 'en'

    def load_language(self, language_code: str) -> bool:
        """
        Load translations for a specific language.

        Args:
            language_code: Language code ('es', 'en', etc.)

        Returns:
            True if language was loaded successfully
        """
        try:
            # Create translator
            translator = gettext.translation(
                'messages',
                localedir=str(self.locale_dir),
                languages=[language_code],
                fallback=True
            )

            # Install translator
            translator.install()

            self.current_translator = translator
            self.current_language = language_code

            logger.info(f"✓ Idioma cargado: {language_code} ({self.supported_languages.get(language_code, 'Desconocido')})")
            return True

        except FileNotFoundError:
            logger.warning(f"⚠ Archivo de traducción no encontrado para: {language_code}")
            # Fallback to English
            if language_code != self.fallback_language:
                return self.load_language(self.fallback_language)
            return False
        except Exception as e:
            logger.error(f"⚠ Error cargando idioma {language_code}: {e}")
            return False

    def get_text(self, message: str) -> str:
        """
        Get translated text for a message.

        Args:
            message: Original message text

        Returns:
            Translated text or original if no translation available
        """
        if self.current_translator:
            return self.current_translator.gettext(message)
        return message

    def get_current_language(self) -> str:
        """Get the current language code."""
        return self.current_language or 'en'

    def get_available_languages(self) -> list:
        """
        Get list of available languages.

        Returns:
            List of language codes with traducciones disponibles
        """
        available = []
        if self.locale_dir.exists():
            for item in self.locale_dir.iterdir():
                if item.is_dir() and (item / 'LC_MESSAGES' / 'messages.mo').exists():
                    available.append(item.name)
        
        # Retornar idiomas soportados que existen en locale_dir
        return [lang for lang in self.supported_languages.keys() if lang in available or lang == 'en']

    def get_language_name(self, language_code: str) -> str:
        """
        Get the full name of a language.

        Args:
            language_code: Language code

        Returns:
            Full language name
        """
        return self.supported_languages.get(language_code, language_code)

    def auto_detect_and_load(self) -> bool:
        """
        Auto-detect system language and load appropriate translations.

        Returns:
            True if language was loaded successfully
        """
        return self.load_language(self.system_language)

    def translate_class_name(self, english_name: str) -> str:
        """
        Translate a class name from English to current language.

        Args:
            english_name: English class name

        Returns:
            Translated class name
        """
        return self.get_text(english_name)

    def format_message(self, template: str, **kwargs) -> str:
        """
        Format a message with variables (similar to f-strings for translations).

        Args:
            template: Template string con placeholders {variable}
            **kwargs: Variables para reemplazar en el template

        Returns:
            Mensaje formateado y traducido
        """
        try:
            translated = self.get_text(template)
            return translated.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Variable de traducción no proporcionada: {e}")
            return template

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get all supported languages.

        Returns:
            Diccionario de {code: nombre}
        """
        return self.supported_languages.copy()

    def add_supported_language(self, language_code: str, language_name: str) -> None:
        """
        Add a new supported language.

        Args:
            language_code: Código del idioma (ej: 'it')
            language_name: Nombre completo del idioma
        """
        self.supported_languages[language_code] = language_name
        logger.info(f"✓ Idioma añadido: {language_code} - {language_name}")


# Global instance
i18n = I18nManager()


def _(message: str) -> str:
    """
    Convenience function for translation.

    Args:
        message: Message to translate

    Returns:
        Translated message
    """
    return i18n.get_text(message)


def get_class_name_translation(english_name: str) -> str:
    """
    Get translated class name.

    Args:
        english_name: English class name

    Returns:
        Translated class name
    """
    return i18n.translate_class_name(english_name)