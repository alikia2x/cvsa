SELECT
    date_trunc('hour', created_at)
        + (floor(extract(minute from created_at) / 10) * 10 || ' minutes')::interval
             AS time_slot,
    COUNT(*) AS record_count
FROM
    video_snapshot
WHERE
    created_at >= NOW() - INTERVAL '48 hours'
  AND created_at <= NOW()
GROUP BY
    time_slot
ORDER BY
    time_slot;