# Modelo de datos Neo4j

## Entities

- Species
    - Propiedades:
        - name: nombre común 
        - scientific_name: nombre científico
        - weight_min_kg: peso mínimo en kilogramos
        - weight_max_kg: peso máximo en kilogramos
        - length_min_m: longitud mínima en metros
        - length_max_m: longitud máxima en metros
        - height_cm: altura en centímetros
        - top_speed_kmh: velocidad máxima en kilómetros por hora
        - gestation_period: duración del periodo de gestación
        - maturity_age: edad a la que dejan de ser crías

    - Relaciones:

        - (:Species)-[:MEMBER_OF_FAMILY]->(:Family): Conecta la especie con su Género, Familia u Orden.
        - (:Species)-[:IS_A]->(:AnimalGroup): Tipo de animal (mamífero, reptil, viviparo ...)
        - (:Species)-[:HAS_SKELETON]->(:SkeletalStructure): Vertebrado o invertebrado
        - (:Species)-[:REPRODUCES_VIA]->(:ReproductionMethod): Ovíparo, Vivíparo, Ovovivíparo
        - (:Species)-[:INHABIT]->(:Habitat): Conecta la especie con su hábitat
        - (:Species)-[:LIVES_IN]->(:Location): Ubicación geográfica (continentes o regiones).
        - (:Species)-[:HAS_ACTIVITY_CYCLE]->(:ActivityCycle)
        - (:Species)-[:ORGANIZED_IN]->(:SocialStructure)
        - (:Species)-[:HAS_DIET_TYPE]->(:DietType)
        - (:Species)-[:PREYS_ON]->(:Species): Un animal es depredador de otro
        - (:Species)-[:FEEDS_ON]->(:FoodSource): Un animal se alimenta de cualquier cosa que no sea un animal (plantas, frutos, semillas, etc.)
        - (:Species)-[:HAS_CONSERVATION_STATUS]->(:ConservationStatus): Estado de conservación
        - (Species)-[:MIGRATES_TO]->(Location):
            Propiedades de la relación: season (Winter, Summer), purpose (Breeding, Feeding)

- AnimalGroup:
    - Propiedades:
        - type: Mammal, Reptile, Bird, Fish, Amphibian.

- SkeletalStructure:
    - Propiedades:
        - type: Vertebrate, Invertebrate

- ReproductionMethod:
    - Propiedades:
        - type: Oviparous, Viviparous, Ovoviviparous

- ConservationStatus:
    - Propiedades:
        - type: Extinct, Extinct in the wild, Critically Endangered, Endangered, Vulnerable, Near Theatened, Least Concerned. Según el IUCN Red List

- ActivityCycle: Momento de actividad del animal
    - Propiedades:
        - type: Nocturnal, Diurnal, Crepuscular

- SocialStructure: Estructura social de la especie
    - Propiedades:
        - type: Solitary, Pack, Herd, Pride

- DietType:
    - Propiedades:
        - type: Carnivore, Herbivore, Omnivore

- FoodSource
    - Propiedades:
        - type: plant, seed ...