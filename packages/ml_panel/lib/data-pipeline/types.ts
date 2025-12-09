// Data pipeline specific types
import type { DatasetRecord, UserLabel, EmbeddingModel, InconsistentLabel } from "../types";

// Database types from packages/core
export interface VideoMetadata {
	aid: number;
	title: string;
	description: string;
	tags: string;
	createdAt?: string;
}

export interface VideoTypeLabel {
	id: number;
	aid: number;
	label: boolean;
	user: string;
	createdAt: string;
}

export interface EmbeddingRecord {
	id: number;
	modelName: string;
	dataChecksum: string;
	vec2048?: number[];
	vec1536?: number[];
	vec1024?: number[];
	createdAt?: string;
}

export interface DataPipelineConfig {
	embeddingModels: EmbeddingModel[];
	batchSize: number;
	requireConsensus: boolean;
	maxInconsistentRatio: number;
}

export interface ProcessedDataset {
	records: DatasetRecord[];
	inconsistentLabels: InconsistentLabel[];
	statistics: {
		totalRecords: number;
		labeledRecords: number;
		inconsistentRecords: number;
		embeddingCoverage: Record<string, number>;
	};
}
