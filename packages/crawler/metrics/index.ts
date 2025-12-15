import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";

const exporter = new OTLPMetricExporter({
	url: "http://localhost:4317",
});

const metricReader = new PeriodicExportingMetricReader({
	exporter: exporter,
	exportIntervalMillis: 2000,
});

const meterProvider = new MeterProvider({
	readers: [metricReader],
});

const meter = meterProvider.getMeter("bullmq-worker");
const anotherMeter = meterProvider.getMeter("networking");

export const ipProxyCounter = anotherMeter.createCounter("ip_proxy_count", {
	description: "Number of requests using IP proxy",
});

export const ipProxyErrorCounter = anotherMeter.createCounter("ip_proxy_error_count", {
	description: "Number of errors thrown by IP proxy",
});

export const aliFCCounter = anotherMeter.createCounter("ali_fc_count", {
	description: "Number of requests using Ali FC",
});

export const aliFCErrorCounter = anotherMeter.createCounter("ali_fc_error_count", {
	description: "Number of errors thrown by Ali FC",
});

export const jobCounter = meter.createCounter("job_count", {
	description: "Number of executed BullMQ jobs",
});

export const queueJobsCounter = meter.createGauge("queue_jobs_count", {
	description: "Number of jobs in specific BullMQ queue",
});

export const jobDurationRaw = meter.createGauge("job_duration_raw", {
	description: "Execution duration of BullMQ jobs in milliseconds",
});

export const snapshotCounter = meter.createCounter("snapshot_count", {
	description: "Number of snapshots taken",
});
