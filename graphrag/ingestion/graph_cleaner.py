# ==========================================
#   GRAPH CLEANER CLASS
# ==========================================

from graphrag.graph.neo4j_manager import Neo4jManager


class GraphCleaner:
    def __init__(
            self,
            neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager

    def clean_graph(self):
        """
        Performs a series of cleaning operations on the graph to ensure data integrity and consistency.
        """

        self._delete_self_loops()
        self._delete_carnivore_foods()
        self._delete_herbivore_preys()
        self._delete_isolated_nodes()

    def _delete_self_loops(self):
        """
        Deletes all self-loop relationships in the graph.
        """
        query = """
        MATCH (n)-[r]->(n)
        WITH type(r) AS rel_type, r
        DELETE r
        RETURN rel_type, count(r) AS deleted_count
        """
        results = self.neo4j.execute_query(query)
        
        for row in results:
            print(f"Deleted {row['deleted_count']} self-loops of the relationship '{row['rel_type']}'")
                
        return results

    
    def _delete_isolated_nodes(self):
        """
        Deletes all nodes that have no relationships.
        """
        query = """
        MATCH (n)
        WHERE NOT (n)--()
        WITH labels(n) AS node_labels, n
        DELETE n
        RETURN node_labels, count(n) AS deleted_count
        """
        results = self.neo4j.execute_query(query)
        
        for row in results:
            print(f"Deleted {row['deleted_count']} isolated nodes with labels: {row['node_labels']}")
                
        return results

    def _delete_carnivore_foods(self):
        """
        Deletes all 'FEEDS_ON' relationships where the source node is a 'Carnivore'.
        """
        query = """
        MATCH (s:Species)-[:HAS_DIET_TYPE]->({type: "Carnivore"})
        MATCH (s)-[r:FEEDS_ON]->()
        WITH type(r) AS rel_type, r
        DELETE r
        RETURN rel_type, count(r) AS deleted_count
        """
        results = self.neo4j.execute_query(query)
        
        for row in results:
            print(f"Deleted {row['deleted_count']} '{row['rel_type']}' relationships for Carnivores.")
                
        return results

    def _delete_herbivore_preys(self):
        """
        Deletes all 'PREYS_ON' relationships where the source node is an 'Herbivore'.
        """
        query = """
        MATCH (s:Species)-[:HAS_DIET_TYPE]->({type: "Herbivore"})
        MATCH (s)-[r:PREYS_ON]->()
        WITH type(r) AS rel_type, r
        DELETE r
        RETURN rel_type, count(r) AS deleted_count
        """
        results = self.neo4j.execute_query(query)
        

        for row in results:
            print(f"Deleted {row['deleted_count']} '{row['rel_type']}' relationships for Herbivores.")
                
        return results
        