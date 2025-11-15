"""
Internationalization (i18n) module for Air Draw Classifier.

Provides multi-language support using gettext for UI text translation.
"""

import gettext
import locale
import os
from pathlib import Path
from typing import Optional


class I18nManager:
    """
    Manager for internationalization support.

    Handles language detection, translation loading, and text translation.
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

            print(f"✓ Language loaded: {language_code}")
            return True

        except FileNotFoundError:
            print(f"⚠ Translation file not found for language: {language_code}")
            # Fallback to English
            if language_code != 'en':
                return self.load_language('en')
            return False
        except Exception as e:
            print(f"⚠ Error loading language {language_code}: {e}")
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
            List of language codes
        """
        available = []
        if self.locale_dir.exists():
            for item in self.locale_dir.iterdir():
                if item.is_dir() and (item / 'LC_MESSAGES' / 'messages.mo').exists():
                    available.append(item.name)
        return available

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