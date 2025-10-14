import os
from notion_client import Client
from dotenv import load_dotenv
# Inicializar cliente con tu token
notion = Client(auth=os.getenv("NOTION_TOKEN"))

# ID de tu base de datos
database_id = os.getenv("BD_ID")

# Consultar registros
response = notion.databases.query(database_id=database_id)

for page in response["results"]:
    props = page["properties"]

    # Evitar error si el campo está vacío
    nombre = props["Nombre"]["title"][0]["plain_text"] if props["Nombre"]["title"] else "(sin nombre)"
    precio = props["Precio"]["number"] or 0

    print(nombre, "-", precio)


notion.pages.create(
    parent={"database_id": database_id},
    properties={
        "Nombre": {"title": [{"text": {"content": "Risotto de Hongos"}}]},
        "Precio": {"number": 22},
        "Categoria": {"select": {"name": "Vegetariano"}}
    }
)
print("✅ Nuevo plato agregado")
