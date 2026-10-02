
import json
from pathlib import Path

from langchain_core.documents import Document

data_dir = Path(__file__).parent.parent / "data"
json_data=[]


def load_json_docs() -> list:
    """Loads json objects from json files and returns them as a list"""
    for file_path in data_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.loads(f.read().lower())
                json_data.extend(data)

                print(f"Loaded {len(json_data)} records from {file_path.name}")
                print(json_data[0])
            
        except json.JSONDecodeError:
            print(f"Error : {file_path.name} contains invalid json")

    return json_data    


def get_semantic_text(recipe):
    """Converts json data into semantic text in natural language"""
    name = recipe['name']

    # 1. format ingredients
    formatted_ingredients = f"{name} is made of the following ingredients: " + ", ".join(recipe['ingredients'])

    # 2. format nutrition info
    formatted_nutrition = f"{name} has the following nutrition : " + ", ".join(f"{k} is {v}" for k, v in recipe['nutrition'].items())

    # 3. combine all formatted parts into the final semantic text
    semantic_text = f"{name} is a {recipe['cuisine']} dish with a cooktime of {recipe['cooktime_minutes']} minutes. {formatted_ingredients}. {formatted_nutrition}."
    
    return semantic_text


def get_metadata(recipe):
    return {
        # "id": recipe['id'],
        "name": recipe['name'],
        "cooktime_minutes": recipe['cooktime_minutes'],
        "cuisine": recipe['cuisine'],
        # "instructions": recipe['instructions']
    }


def get_json_data_as_semantic_text_document()->list[Document]:
    """Converts all loaded json data into semantic text documents and returns them as a list of Document objects."""
    
    json_data = load_json_docs()
    semantic_data = []

    for recipe in json_data:
        semantic_text = get_semantic_text(recipe)
        metadata = get_metadata(recipe)
        semantic_data.append(Document(page_content=semantic_text, metadata=metadata))

    return semantic_data


if __name__ == "__main__":
    load_json_docs()