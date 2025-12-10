// API types for ML training service

export interface EmbeddingModel {
	name: string;
	dimensions: number;
	type: string;
	api_endpoint?: string;
	max_tokens?: number;
	max_batch_size?: number;
}

export interface EmbeddingModelsResponse {
	models: Record<string, EmbeddingModel>;
}

export interface DatasetStats {
	total_records: number;
	new_embeddings: number;
	reused_embeddings: number;
	inconsistent_labels: number;
	embedding_model: string;
}

export interface Dataset {
	dataset_id: string;
	description?: string;
	stats: DatasetStats;
	created_at: string;
}

export interface DatasetsResponse {
	datasets: Dataset[];
}

export interface DatasetDetail {
	dataset_id: string;
	dataset: any[];
	description?: string;
	stats: DatasetStats;
	created_at: string;
}

export interface SamplingRequest {
	strategy: string;
	limit?: number;
}

export interface SamplingResponse {
	strategy: string;
	total_available: number;
	sampled_count: number;
	aid_list: number[];
	filters_applied?: Record<string, any>;
	sampling_info: Record<string, any>;
}

export interface DatasetCreateRequest {
	sampling: SamplingRequest;
	embedding_model: string;
	force_regenerate?: boolean;
	description?: string;
}

export interface DatasetCreateResponse {
	dataset_id: string;
	sampling_response: SamplingResponse;
	task_id: string;
	total_records: number;
	status: string;
	message: string;
	description?: string;
}

export interface TaskProgress {
	current_step: string;
	total_steps: number;
	completed_steps: number;
	percentage: number;
	message?: string;
	estimated_time_remaining?: number;
}

export interface Task {
	task_id: string;
	status: string;
	progress?: TaskProgress;
	result?: Record<string, any>;
	error?: string;
	created_at: string;
	started_at?: string;
	completed_at?: string;
}

export interface TasksResponse {
	tasks: Task[];
	total_count: number;
	pending_count: number;
	running_count: number;
	completed_count: number;
	failed_count: number;
}

export interface SamplingStats {
	total_labeled_videos: number;
	positive_labels: number;
	negative_labels: number;
}

export interface DatasetStatistics {
	total_datasets: number;
	valid_datasets: number;
	error_datasets: number;
	total_records: number;
	total_new_embeddings: number;
	total_reused_embeddings: number;
	storage_directory: string;
}

export interface HealthResponse {
	status: string;
	service: string;
	embedding_service: any;
	database: string;
	available_models: string[];
}
