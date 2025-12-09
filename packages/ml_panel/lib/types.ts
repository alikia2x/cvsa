// Shared ML types and interfaces
export interface DatasetRecord {
	aid: number;
	title: string;
	description: string;
	tags: string;
	embedding?: number[];
	label?: boolean;
	userLabels?: UserLabel[];
}

export interface UserLabel {
	user: string;
	label: boolean;
	createdAt: string;
}

export interface EmbeddingModel {
	name: string;
	dimensions: number;
	type: "openai-compatible" | "local";
	apiEndpoint?: string;
}

export interface TrainingConfig {
	learningRate: number;
	batchSize: number;
	epochs: number;
	earlyStop: boolean;
	patience?: number;
	embeddingModel: string;
}

export interface ExperimentResult {
	experimentId: string;
	config: TrainingConfig;
	metrics: {
		accuracy: number;
		precision: number;
		recall: number;
		f1: number;
	};
	createdAt: string;
	status: "running" | "completed" | "failed";
}

export interface InconsistentLabel {
	aid: number;
	title: string;
	description: string;
	tags: string;
	labels: UserLabel[];
	consensus?: boolean;
}
