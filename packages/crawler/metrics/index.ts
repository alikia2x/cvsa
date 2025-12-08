import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";

const exporter = new OTLPMetricExporter({
	url: "http://localhost:4317"
});

const metricReader = new PeriodicExportingMetricReader({
	exporter: exporter,
	exportIntervalMillis: 2000
});

const meterProvider = new MeterProvider({
	readers: [metricReader]
});

const meter = meterProvider.getMeter("bullmq-worker");

export const jobCounter = meter.createCounter("job_count", {
	description: "Number of executed BullMQ jobs"
});

export const queueJobsCounter = meter.createGauge("queue_jobs_count", {
	description: "Number of jobs in specific BullMQ queue"
});

export const jobDurationRaw = meter.createGauge("job_duration_raw", {
	description: "Execution duration of BullMQ jobs in milliseconds"
});

export const snapshotCounter = meter.createCounter("snapshot_count", {
	description: "Number of snapshots taken"
});