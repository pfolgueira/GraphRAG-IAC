import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("../../env/.env")

sys.path.append('../..')
from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.ingestion.text_processor import TextProcessor

DATA_DIR = Path("../../scraping/data/")
PROCESSED_FILE_PATH = Path("./processed_animals.txt")

initial_animals = [
    "Tiger", "Crocodile", "Lion", "Elephant", "Dolphin", "Giant Panda", "Horse", "Penguin", "Wolf", "Shark", "Rabbit",
    "Gorilla", "Giraffe", "Cheetah", "Polar Bear", "Hippopotamus", "Zebra", "Red Panda", "Kangaroo", "Koala", "Sloth",
    "Meerkat", "Rhinoceros", "Snow Leopard", "Orangutan", "Chimpanzee", "Platypus", "Lemur", "Capybara", "Sea Otter", "Arctic Fox",
    "Jaguar", "Black Panther", "Brown Bear", "Okapi", "Tasmanian Devil", "Wombat", "Quokka", "Hedgehog", "Fennec Fox", "Bald Eagle",
    "Peacock", "Flamingo", "Parrot", "Hummingbird", "Swan", "Toucan", "Owl", "Puffin", "Woodpecker", "Raven",
    "Falcon", "Ostrich", "Emu", "Albatross", "Robin", "Blue Jay", "Cardinal", "Canary", "Guinea Pig", "Hamster",
    "Cow", "Pig", "Sheep", "Goat", "Chicken", "Duck", "Donkey", "Llama", "Alpaca", "Ferret",
    "Chinchilla", "Great White Shark", "Blue Whale", "Killer Whale (Orca)", "Sea Turtle", "Octopus", "Seahorse", "Clownfish", "Axolotl", "Manatee",
    "Stingray", "Walrus", "Seal", "Narwhal", "Lobster", "Jellyfish", "Chameleon", "Komodo Dragon", "King Cobra", "Bearded Dragon",
    "Green Iguana", "Tree Frog", "Monarch Butterfly", "Honey Bee", "Praying Mantis", "Ladybug", "Raccoon", "Camel", "Squirrel", "Buffalo"
]

def initialize_neo4j():
    """Initialize Neo4j connection, create constraints and vector index."""
    neo4j_manager = Neo4jManager()
    neo4j_manager.create_constraints()
    neo4j_manager.create_vector_index()

    return neo4j_manager

def update_animals_list(neo4j_manager: Neo4jManager):
    """Update initial animal list with animals that have already been added to the database."""
    query = "MATCH (s:Species) RETURN s.name AS name"

    query = """
    MATCH (s:Species)
    WHERE NOT s.name IN $lista_python
    RETURN s.name AS new_species
    """
    results = neo4j_manager.execute_query(query, {"lista_python": initial_animals})
    animals = [record['new_species'] for record in results]
    initial_animals.extend(animals)

def load_document(file_path:str):
    """Load a document"""
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    return markdown_content

def main():
    neo4j_manager = initialize_neo4j()
    update_animals_list(neo4j_manager)
    
    text_processor = TextProcessor(neo4j_manager, species_names=initial_animals, chunk_size=800, chunk_overlap=80)

    processed_animals = set()
    if PROCESSED_FILE_PATH.exists():
        with open(PROCESSED_FILE_PATH, 'r', encoding='utf-8') as f:
            processed_animals = {line.strip() for line in f}

    files = list(DATA_DIR.glob('*.md'))
    pending_animals = [f for f in files if f.name not in processed_animals]

    print(f"Total de archivos: {len(files)} | Procesados: {len(processed_animals)} | Pendientes: {len(pending_animals)}")

    files_to_process = 1
    
    for file in pending_animals[:files_to_process]:
        print(f"Procesando {file.name}")
        
        try:
            md_content = load_document(str(file))
            animal = file.stem 

            fomatted_name = animal.replace('_', ' ').replace('-', ' ').title()

            text_processor.process_document(
                text=md_content, 
                document_id=animal,
                metadata={"title": fomatted_name, "source": "A-Z Animals"}
            )

            with open(PROCESSED_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(file.name + '\n')
                
            
        except Exception as e:
            print(f"Error al procesar {file.name}: {e}")
            break

if __name__ == "__main__":
    main()
    