import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";

const exporter = new OTLPMetricExporter({
	url: "http://localhost:4317"
});

const metricReader = new PeriodicExportingMetricReader({
	exporter: exporter,
	exportIntervalMillis: 60000
});

const meterProvider = new MeterProvider({
	readers: [metricReader]
});

const meter = meterProvider.getMeter("bullmq-worker");

export const jobCounter = meter.createCounter("bullmq_job_executed_total", {
	description: "Number of executed BullMQ jobs"
});

export const jobDuration = meter.createHistogram("bullmq_job_duration_ms", {
	description: "Execution duration of BullMQ jobs in milliseconds",
	unit: "ms"
});
