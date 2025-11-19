# Guía de Configuración - Pictionary Live

Esta guía explica cómo configurar el entorno de desarrollo para Pictionary Live usando un entorno virtual (venv), especialmente útil para trabajo colaborativo con Git.

## ¿Por qué usar un entorno virtual?

Un entorno virtual (venv) proporciona:
- **Aislamiento de dependencias**: Cada proyecto tiene sus propias versiones de paquetes
- **Reproducibilidad**: Todos los colaboradores usan las mismas versiones
- **Sin conflictos**: No interfiere con otros proyectos Python en tu sistema
- **Compatible con Git**: El entorno virtual no se sube al repositorio (está en `.gitignore`)

## Requisitos Previos

- **Python 3.10, 3.11 o 3.12** (recomendado para MediaPipe)
  - Python 3.13+ funcionará en modo mouse (sin detección de manos)
- **Git** instalado y configurado
- En Windows: Visual C++ Redistributable puede ser necesario para algunos paquetes

## Configuración Inicial

### Windows

1. **Abrir terminal en la carpeta del proyecto**
   ```cmd
   cd d:\Tarea\4. INTELIGENCIA ARTIFICIAL\.TPs\IA-Proyecto
   ```

2. **Ejecutar el script de configuración**
   ```cmd
   setup.bat
   ```
   
   Esto automáticamente:
   - Verifica tu versión de Python
   - Crea un entorno virtual en `venv/`
   - Instala todas las dependencias desde `requirements.txt`

3. **Ejecutar la aplicación**
   ```cmd
   run.bat
   ```
   
   O manualmente:
   ```cmd
   activate.bat
   python main.py
   ```

### Linux / macOS

1. **Abrir terminal en la carpeta del proyecto**
   ```bash
   cd ~/proyectos/IA-Proyecto
   ```

2. **Hacer ejecutables los scripts** (solo la primera vez)
   ```bash
   chmod +x setup.sh run.sh
   ```

3. **Ejecutar el script de configuración**
   ```bash
   ./setup.sh
   ```
   
   Esto automáticamente:
   - Verifica tu versión de Python
   - Crea un entorno virtual en `venv/`
   - Instala todas las dependencias desde `requirements.txt`

4. **Ejecutar la aplicación**
   ```bash
   ./run.sh
   ```
   
   O manualmente:
   ```bash
   source venv/bin/activate
   python main.py
   ```

## Uso Diario

### Activar el entorno virtual

**Windows:**
```cmd
venv\Scripts\activate.bat
```
O simplemente ejecuta `activate.bat`

**Linux/Mac:**
```bash
source venv/bin/activate
```

Verás `(venv)` al inicio de tu prompt cuando esté activado.

### Desactivar el entorno virtual

Simplemente ejecuta:
```bash
deactivate
```

### Ejecutar la aplicación

Con el entorno activado:
```bash
python main.py
```

O usa los scripts convenientes:
- Windows: `run.bat`
- Linux/Mac: `./run.sh`

## Gestión de Dependencias

### Ver paquetes instalados

```bash
pip list
```

### Instalar un nuevo paquete

```bash
pip install nombre-paquete
```

### Actualizar requirements.txt

Después de instalar nuevos paquetes:
```bash
pip freeze > requirements.txt
```

**Importante**: Antes de hacer commit, revisa que `requirements.txt` solo incluya dependencias necesarias.

### Actualizar dependencias existentes

```bash
pip install --upgrade -r requirements.txt
```

## Trabajo Colaborativo con Git

### Para nuevos colaboradores

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/fedemuntaabski/IA-Proyecto.git
   cd IA-Proyecto
   ```

2. **Configurar el entorno**
   - Windows: `setup.bat`
   - Linux/Mac: `./setup.sh`

3. **Empezar a trabajar**
   - Windows: `run.bat`
   - Linux/Mac: `./run.sh`

### El directorio venv/ está en .gitignore

El directorio `venv/` **NO** se sube a Git. Esto significa:
- ✅ Cada colaborador crea su propio entorno virtual
- ✅ No hay archivos binarios grandes en el repositorio
- ✅ Funciona en diferentes sistemas operativos
- ✅ El archivo `requirements.txt` asegura que todos usen las mismas versiones

### Cuando alguien actualiza requirements.txt

Si ves cambios en `requirements.txt` después de hacer `git pull`:

```bash
# Activar el entorno virtual primero
# Luego actualizar los paquetes
pip install -r requirements.txt
```

## Solución de Problemas

### Error: "Python no encontrado"

**Verifica la instalación:**
```bash
python --version
# o
python3 --version
```

Si no está instalado, descarga desde [python.org](https://python.org/downloads/).

### Error: "No module named 'venv'"

En algunas distribuciones Linux, necesitas instalar el paquete venv:
```bash
# Ubuntu/Debian
sudo apt-get install python3-venv

# Fedora
sudo dnf install python3-venv
```

### Error: TensorFlow no disponible

Esto puede pasar si:
1. **Python 3.13+**: TensorFlow no está totalmente soportado aún. Usa Python 3.10-3.12
2. **Instalación fallida**: Intenta reinstalar:
   ```bash
   pip install --force-reinstall tensorflow
   ```

### Advertencia: MediaPipe no disponible

MediaPipe solo funciona en Python 3.10-3.12. Si usas Python 3.13+:
- La aplicación funcionará en **modo mouse**
- No habrá detección de manos
- Para usar detección de manos, instala Python 3.10, 3.11 o 3.12

### El entorno virtual no se activa

**Windows:**
- Si ves un error de políticas de ejecución:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  Luego intenta nuevamente.

**Linux/Mac:**
- Asegúrate de usar `source` antes del path:
  ```bash
  source venv/bin/activate
  ```

### Conflictos de versiones

Si tienes problemas con versiones de paquetes:

1. **Eliminar el entorno virtual**
   ```bash
   # Windows
   rmdir /s /q venv
   
   # Linux/Mac
   rm -rf venv
   ```

2. **Recrear desde cero**
   ```bash
   # Windows
   setup.bat
   
   # Linux/Mac
   ./setup.sh
   ```

## Estructura de Archivos de Configuración

```
IA-Proyecto/
├── venv/                    # Entorno virtual (NO en Git)
├── requirements.txt         # Dependencias del proyecto
├── setup.bat               # Script de configuración (Windows)
├── setup.sh                # Script de configuración (Linux/Mac)
├── run.bat                 # Script de ejecución (Windows)
├── run.sh                  # Script de ejecución (Linux/Mac)
├── activate.bat            # Activar venv manualmente (Windows)
├── .gitignore              # Incluye venv/ y otros archivos temporales
└── SETUP.md               # Esta guía
```

## Buenas Prácticas

1. **Siempre activa el entorno virtual** antes de trabajar en el proyecto
2. **No modifiques archivos dentro de `venv/`** - usa `pip` para gestionar paquetes
3. **Actualiza `requirements.txt`** si instalas nuevas dependencias
4. **Haz commit de `requirements.txt`** cuando cambien las dependencias
5. **NO hagas commit de `venv/`** - ya está en `.gitignore`
6. **Comunica a tu equipo** cuando actualices dependencias

## Comandos Útiles

```bash
# Ver información del entorno virtual
pip show pip

# Verificar paquetes desactualizados
pip list --outdated

# Buscar un paquete específico
pip show nombre-paquete

# Ver dependencias de un paquete
pip show nombre-paquete | grep Requires

# Desinstalar un paquete
pip uninstall nombre-paquete

# Instalar desde una versión específica
pip install paquete==1.2.3
```

## Recursos Adicionales

- [Documentación oficial de venv](https://docs.python.org/3/library/venv.html)
- [Guía de pip](https://pip.pypa.io/en/stable/user_guide/)
- [README del proyecto](README.md)
- [Documentación PyQt6](README_PYQT6.md)

## Soporte

Si encuentras problemas no cubiertos en esta guía:
1. Revisa los mensajes de error cuidadosamente
2. Busca en Google el mensaje de error completo
3. Consulta con tu equipo o mentor
4. Crea un issue en el repositorio de GitHub
