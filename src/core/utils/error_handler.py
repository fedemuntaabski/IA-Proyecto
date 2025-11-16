"""
Error Handler - Gestor Mejorado de Errores.

Este módulo proporciona mensajes de error más claros y útiles,
con sugerencias para resolver problemas comunes.
"""

from typing import Optional, Tuple
from ..i18n import _


class ErrorMessages:
    """Proporciona mensajes de error mejorados y claros."""

    # Mensajes de error con emojis y sugerencias
    MESSAGES = {
        'camera_not_found': {
            'title': '❌ No se encontró cámara',
            'message': 'La aplicación no pudo acceder a la cámara del dispositivo.',
            'suggestions': [
                'Verifica que la cámara está conectada',
                'Comprueba los permisos de la aplicación',
                'Intenta reiniciar la aplicación'
            ]
        },
        'model_not_found': {
            'title': '❌ Modelo no encontrado',
            'message': 'No se pudo cargar el modelo de clasificación.',
            'suggestions': [
                'Verifica que el archivo del modelo existe en la carpeta IA/',
                'Descarga el modelo desde el repositorio',
                'Comprueba la integridad del archivo'
            ]
        },
        'gpu_initialization_failed': {
            'title': '⚠️  Error en aceleración GPU',
            'message': 'No se pudo inicializar la GPU. La aplicación usará CPU.',
            'suggestions': [
                'Actualiza los drivers de tu tarjeta gráfica',
                'Verifica que CUDA está correctamente instalado',
                'La aplicación continuará funcionando en CPU'
            ]
        },
        'inference_failed': {
            'title': '❌ Error en clasificación',
            'message': 'No se pudo clasificar el dibujo.',
            'suggestions': [
                'Intenta dibujar de nuevo',
                'Dibuja con más claridad',
                'Asegúrate de que hay buena iluminación'
            ]
        },
        'invalid_configuration': {
            'title': '❌ Configuración inválida',
            'message': 'Hay errores en la configuración de la aplicación.',
            'suggestions': [
                'Revisa los archivos de configuración',
                'Restaura la configuración por defecto',
                'Contacta con soporte si el problema persiste'
            ]
        },
        'permission_denied': {
            'title': '❌ Permiso denegado',
            'message': 'La aplicación no tiene permisos para acceder a recursos necesarios.',
            'suggestions': [
                'Otorga permisos de cámara a la aplicación',
                'Verifica la configuración de seguridad del sistema',
                'Reinicia la aplicación después de otorgar permisos'
            ]
        },
        'out_of_memory': {
            'title': '⚠️  Memoria insuficiente',
            'message': 'La aplicación está usando demasiada memoria.',
            'suggestions': [
                'Cierra otras aplicaciones',
                'Libera memoria del sistema',
                'Reinicia la aplicación'
            ]
        },
        'file_access_error': {
            'title': '❌ Error de acceso a archivo',
            'message': 'No se pudo acceder a un archivo necesario.',
            'suggestions': [
                'Verifica que el archivo existe',
                'Comprueba los permisos de carpeta',
                'Intenta mover la aplicación a una carpeta diferente'
            ]
        }
    }

    @classmethod
    def get_error_message(cls, error_type: str) -> Tuple[str, str, list]:
        """
        Obtiene un mensaje de error estructurado.

        Args:
            error_type: Tipo de error

        Returns:
            Tupla de (título, mensaje, sugerencias)
        """
        if error_type not in cls.MESSAGES:
            return (
                '❌ Error desconocido',
                'Ocurrió un error inesperado en la aplicación.',
                ['Intenta reiniciar la aplicación', 'Contacta con soporte si el problema persiste']
            )

        msg = cls.MESSAGES[error_type]
        return msg['title'], msg['message'], msg['suggestions']

    @classmethod
    def print_error(cls, error_type: str, detailed_message: Optional[str] = None) -> None:
        """
        Imprime un mensaje de error formateado.

        Args:
            error_type: Tipo de error
            detailed_message: Mensaje de error detallado opcional
        """
        title, message, suggestions = cls.get_error_message(error_type)

        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print(message)

        if detailed_message:
            print(f"\n📋 {_('Detalles')}: {detailed_message}")

        print(f"\n💡 {_('Sugerencias')}:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")

        print("=" * 60 + "\n")

    @classmethod
    def print_warning(cls, title: str, message: str, suggestions: Optional[list] = None) -> None:
        """
        Imprime un mensaje de advertencia formateado.

        Args:
            title: Título de la advertencia
            message: Mensaje de advertencia
            suggestions: Lista de sugerencias opcionales
        """
        print("\n" + "=" * 60)
        print(f"⚠️  {title}")
        print("=" * 60)
        print(message)

        if suggestions:
            print(f"\n💡 {_('Sugerencias')}:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")

        print("=" * 60 + "\n")

    @classmethod
    def print_success(cls, title: str, message: str) -> None:
        """
        Imprime un mensaje de éxito formateado.

        Args:
            title: Título del mensaje
            message: Mensaje de éxito
        """
        print("\n" + "=" * 60)
        print(f"✅ {title}")
        print("=" * 60)
        print(message)
        print("=" * 60 + "\n")


# Instancia global para conveniencia
def handle_error(error_type: str, detailed_message: Optional[str] = None) -> None:
    """
    Función global para manejar errores de forma consistente.

    Args:
        error_type: Tipo de error
        detailed_message: Mensaje detallado opcional
    """
    ErrorMessages.print_error(error_type, detailed_message)


def handle_warning(title: str, message: str, suggestions: Optional[list] = None) -> None:
    """
    Función global para manejar advertencias.

    Args:
        title: Título de la advertencia
        message: Mensaje
        suggestions: Sugerencias opcionales
    """
    ErrorMessages.print_warning(title, message, suggestions)


def handle_success(title: str, message: str) -> None:
    """
    Función global para mostrar mensajes de éxito.

    Args:
        title: Título del mensaje
        message: Mensaje
    """
    ErrorMessages.print_success(title, message)
