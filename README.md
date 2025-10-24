<p align="center">
  <img width="577" height="53" alt="Banner del Proyecto" src="https://github.com/user-attachments/assets/a4776e69-c52f-42d3-be06-165cefd12198" />
</p>

<h1 align="center">
  Repositorio del Proyecto Integrador
</h1>

## 👥 Integrantes

* Simon Martin Sposito
* Laura Velazquez
* Facundo Jimenez
* Matias Torres

## 🚀 Inicio Rápido

> **Nota Importante:** El proyecto fue desarrollado en Google Colab, por lo que recomendamos probarlo en dicha herramienta. Puede ser testeado localmente haciendo unas leves modificaciones.

---

Sigue estos pasos para poner en marcha el agente de la ferretería **"Proyecto\_Agente\_CACIC.py"**.

### 1. Instalación

Instala todas las librerías necesarias. La forma más fácil es usando el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
````

### 2\. Configuración de Claves (API Keys)

El agente necesita varias claves de API para funcionar. Añádelas a tu codigo/entorno (en Colab se hace desde la pestaña "Secrets" 🔑).

| Servicio | Dónde Encontrarla |
| :--- | :--- |
| **Notion DB ID** | ID de tu Base de Datos de Notion. (Crea una BD y conecta tu integración). |
| **Notion API Token** | [developers.notion.com](https://developers.notion.com) |
| **Google AI API Key** | [aistudio.google.com](https://aistudio.google.com/) |
| **LangSmith API Key** | [smith.langchain.com](https://smith.langchain.com) |
| **Tavily API Key** | [app.tavily.com/home](https://app.tavily.com/home) |
| **LangSmith Project Name**| (Opcional) Úsalo si no quieres que el seguimiento se registre en el proyecto `default`. |

### 💻 Nota sobre Pruebas Locales

> En caso de estar probando localmente, asegúrate de **comentar o eliminar** todas las líneas que contengan `!pip install ...` en el script o notebook.
