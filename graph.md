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
        - (:Species)-[:INHABIT]->(:Habitat): Conecta la especie con su hábitat
        - (:Species)-[:LIVES_IN]->(:Location): Ubicación geográfica (continentes o regiones).

        - activity_cycle: Nocturnal, Diurnal, Crepuscular. Momento de actividad del animal
        - social_structure: Solitary, Pack, Herd, Pride. Estrcutura social de la especie


