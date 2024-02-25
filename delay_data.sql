WITH AggregatedMetrics AS (
  SELECT
      DATEPART(DAY, CAST(m.end_date AS DATE)) AS DayOfMonth,
      DATEDIFF(day, CAST(m.update_date AS DATE), CAST(m.end_date AS DATE)) AS MetricJ_Delay,
      COUNT(m.update_date) AS UpdateCount,
      SUM(m.metric_value) AS MetricValueSum
  FROM
      db.metric_data m
  WHERE
      m.id IN (1001, 1002, 1011)
      AND m.session_end_date >= '2023-01-01'
  GROUP BY
      DATEPART(DAY, CAST(m.end_date AS DATE)),
      DATEDIFF(day, CAST(m.update_date AS DATE), CAST(m.end_date AS DATE))
), CalculatedData AS (
SELECT
    DayOfMonth,
    MetricJ_Delay,
    CAST(UpdateCount AS FLOAT) / SUM(UpdateCount) OVER (PARTITION BY DayOfMonth) AS MetricJ_Proportion,
    CAST(MetricValueSum AS FLOAT) / SUM(MetricValueSum) OVER (PARTITION BY DayOfMonth) AS MetricI_Proportion,
    COUNT(*) OVER (PARTITION BY DayOfMonth, MetricJ_Delay)
FROM
    AggregatedMetrics
ORDER BY
    DayOfMonth,
    MetricJ_Delay
)
-- Main query
SELECT
    DayOfMonth,
    MetricJ_Delay,
    MetricJ_Proportion,
    SUM(MetricJ_Proportion) OVER (PARTITION BY DayOfMonth ORDER BY MetricJ_Delay DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS Cumulative_MetricJ_Proportion,
    MetricI_Proportion,
    SUM(MetricI_Proportion) OVER (PARTITION BY DayOfMonth ORDER BY MetricJ_Delay DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS Cumulative_MetricI_Proportion
FROM
    CalculatedData
ORDER BY
    DayOfMonth, MetricJ_Delay;
