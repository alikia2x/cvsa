SET TIME ZONE 'Asia/Shanghai';
SELECT
    date_trunc('hour', started_at) AS time_slot,
    type,
    COUNT(*) AS record_count
FROM
    snapshot_schedule
WHERE
    started_at <= NOW() + INTERVAL '24 hours'
  AND started_at >= NOW()
GROUP BY
    type, time_slot
ORDER BY
    type, time_slot