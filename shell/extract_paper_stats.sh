#!/bin/bash
# Extract key statistics from evaluation results for paper

echo "=============================================="
echo "FingerPrint² Paper Statistics Extraction"
echo "=============================================="
echo ""

# Find all result databases
DBS=$(find results -name "*.db" -type f 2>/dev/null)

if [ -z "$DBS" ]; then
    echo "No result databases found."
    exit 1
fi

echo "=== Models Evaluated ==="
for db in $DBS; do
    model=$(basename "$db" .db | sed 's/_35k$//' | sed 's/gpu[0-9]_//')
    count=$(sqlite3 "$db" "SELECT COUNT(*) FROM probe_results;" 2>/dev/null || echo "0")
    images=$(sqlite3 "$db" "SELECT COUNT(DISTINCT image_id) FROM probe_results;" 2>/dev/null || echo "0")

    if [ "$count" -gt "0" ]; then
        echo "$model: $count probe results, $images unique images"
    fi
done

echo ""
echo "=== Regional Breakdown (per model) ==="
for db in $DBS; do
    model=$(basename "$db" .db)
    count=$(sqlite3 "$db" "SELECT COUNT(*) FROM probe_results;" 2>/dev/null || echo "0")

    if [ "$count" -gt "1000" ]; then
        echo ""
        echo "Model: $model"
        sqlite3 "$db" "
        SELECT
            jurisdiction_region,
            COUNT(DISTINCT image_id) as images,
            COUNT(*) as probes,
            ROUND(AVG(CASE WHEN response LIKE '%[ERROR]%' THEN 0 ELSE 1 END) * 100, 1) as success_rate
        FROM probe_results
        WHERE jurisdiction_region IS NOT NULL AND jurisdiction_region != 'unknown'
        GROUP BY jurisdiction_region
        ORDER BY images DESC;
        " 2>/dev/null
    fi
done

echo ""
echo "=== Probe Distribution ==="
for db in $DBS; do
    count=$(sqlite3 "$db" "SELECT COUNT(*) FROM probe_results;" 2>/dev/null || echo "0")

    if [ "$count" -gt "1000" ]; then
        model=$(basename "$db" .db | cut -c1-30)
        echo ""
        echo "Model: $model..."
        sqlite3 "$db" "
        SELECT probe_id, COUNT(*) as count
        FROM probe_results
        GROUP BY probe_id
        ORDER BY probe_id;
        " 2>/dev/null
    fi
done

echo ""
echo "=============================================="
