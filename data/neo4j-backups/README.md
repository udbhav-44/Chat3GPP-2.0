# Neo4j Backups

Store consistent backups here; do **not** copy a live database directory.

Recommended commands (run on the Neo4j host):
- Online backup: `neo4j-admin backup --backup-dir <this-dir> --database neo4j`
- Offline dump (stop Neo4j first): `neo4j-admin database dump neo4j --to-path <this-dir>`

Restore into a new deployment with `neo4j-admin database load neo4j --from-path <backup>`.

Keep credentials/URIs in your environment vars; do not commit backup files.
