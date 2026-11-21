# Shopping Sessions: Rust pipeline

Ibrahim Sadat's entry for the Databricks Shopping Sessions track. A single Rust binary that implements `rules.md`:
parallel chunked CSV parsing, hash dedup by `event_id`, union-find over linked IDs, bucketed sessionization,
attribution and assists, and Parquet output.

Build: `docker build -t shopping-sessions .`
