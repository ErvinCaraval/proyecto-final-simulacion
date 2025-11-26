# Proyecto Python: Entorno Virtual y Manejo de Archivos

Este proyecto incluye instrucciones para **crear y usar un entorno virtual en Linux** y buenas prácticas para **evitar subir archivos innecesarios** a GitHub, como imágenes `.png` y `.gif`.

---

## 1️⃣ Crear el entorno virtual

Abre la terminal y navega a la carpeta de tu proyecto:

```bash
cd /ruta/a/tu/proyecto
```

Luego, crea el entorno virtual:

```bash
python3 -m venv venv
```

    Esto creará una carpeta llamada `venv` en tu proyecto, donde se almacenarán todos los paquetes que instales.
    Para activar el entorno virtual, ejecuta:
    
    ```bash
    source venv/bin/activate
    ```
    
    Para desactivar el entorno virtual, ejecuta:
    
    ```bash
    deactivate
    ```
---

## 2️⃣ Instalar paquetes

Para instalar paquetes en el entorno virtual, usa `pip`:

```bash
pip install nombre-del-paquete
```
Para guardar las dependencias instaladas en un archivo `requirements.txt`, ejecuta:

```bash
pip freeze > requirements.txt
```
Para instalar las dependencias desde `requirements.txt`, usa:

```bash
pip install -r requirements.txt
```
---

## 3️⃣ Manejo de archivos

Para evitar subir archivos innecesarios a GitHub, como imágenes `.png` y `.gif`, es recomendable crear un archivo `.gitignore` en la raíz de tu proyecto.
Crea el archivo `.gitignore` y agrega las siguientes líneas:

```plaintext
venv/
*.pyc
__pycache__/
*.png
*.gif
*.jpg
*.jpeg
*.svg
*.ico
*.ipynb
*.ipynb_checkpoints
```
