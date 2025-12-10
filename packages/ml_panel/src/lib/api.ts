// API client for ML training service
import type {
	HealthResponse,
	EmbeddingModelsResponse,
	DatasetsResponse,
	DatasetDetail,
	SamplingStats,
	SamplingResponse,
	DatasetCreateResponse,
	Task,
	TasksResponse,
	DatasetStatistics
} from "@/types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/v1";

class ApiClient {
	private baseUrl: string;

	constructor(baseUrl: string = API_BASE_URL) {
		this.baseUrl = baseUrl;
	}

	private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
		const url = `${this.baseUrl}${endpoint}`;

		const response = await fetch(url, {
			headers: {
				"Content-Type": "application/json",
				...options?.headers
			},
			...options
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
		}

		return response.json();
	}

	// Health check
	async healthCheck(): Promise<HealthResponse> {
		return this.request("/health");
	}

	// Embedding models
	async getEmbeddingModels(): Promise<EmbeddingModelsResponse> {
		return this.request("/models/embedding");
	}

	// Dataset operations
	async getDatasets(): Promise<DatasetsResponse> {
		return this.request("/datasets");
	}

	async getDataset(datasetId: string): Promise<DatasetDetail> {
		return this.request(`/dataset/${datasetId}`);
	}

	async deleteDataset(datasetId: string): Promise<any> {
		return this.request(`/dataset/${datasetId}`, {
			method: "DELETE"
		});
	}

	async buildDataset(data: {
		aid_list: number[];
		embedding_model: string;
		force_regenerate?: boolean;
		description?: string;
	}): Promise<any> {
		return this.request("/dataset/build", {
			method: "POST",
			body: JSON.stringify(data)
		});
	}


	async sampleDataset(data: {
		strategy: string;
		limit?: number;
		label_value?: boolean;
		metadata_filter?: Record<string, any>;
	}): Promise<SamplingResponse> {
		return this.request("/dataset/sample", {
			method: "POST",
			body: JSON.stringify(data)
		});
	}

	async createDatasetWithSampling(data: {
		sampling: {
			strategy: string;
			limit?: number;
			label_value?: boolean;
			metadata_filter?: Record<string, any>;
		};
		embedding_model: string;
		force_regenerate?: boolean;
		description?: string;
	}): Promise<DatasetCreateResponse> {
		return this.request("/dataset/create-with-sampling", {
			method: "POST",
			body: JSON.stringify(data)
		});
	}

	// Task operations
	async getTask(taskId: string): Promise<Task> {
		return this.request(`/tasks/${taskId}`);
	}

	async getTasks(status?: string, limit: number = 50): Promise<TasksResponse> {
		const params = new URLSearchParams();
		if (status) params.append("status", status);
		params.append("limit", limit.toString());

		return this.request(`/tasks?${params.toString()}`);
	}

	async cleanupTasks(maxAgeHours: number = 24): Promise<any> {
		return this.request("/tasks/cleanup", {
			method: "POST",
			body: JSON.stringify({ max_age_hours: maxAgeHours })
		});
	}

	// Dataset statistics
	async getDatasetStatistics(): Promise<DatasetStatistics> {
		return this.request("/datasets/stats");
	}

	async cleanupDatasets(maxAgeDays: number = 30): Promise<any> {
		return this.request("/datasets/cleanup", {
			method: "POST",
			body: JSON.stringify({ max_age_days: maxAgeDays })
		});
	}
}

export const apiClient = new ApiClient();
export default apiClient;
