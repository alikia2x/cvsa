SET TIME ZONE 'Asia/Shanghai';
SELECT
    date_trunc('hour', started_at) +
    (EXTRACT(minute FROM started_at)::int / 5 * INTERVAL '5 minutes') AS window_start,
    COUNT(*) AS count
FROM snapshot_schedule
WHERE started_at >= NOW() - INTERVAL '1 hours' AND status != 'completed' AND started_at <= NOW() + INTERVAL '14 days'
GROUP BY 1
ORDER BY window_start