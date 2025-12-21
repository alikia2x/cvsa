import logger from "@core/log";
import { WorkerError } from "mq/schema";
import apiManager from "./api_manager";

class AkariAPI {
	private readonly serviceReady: Promise<boolean>;

	constructor() {
		// Wait for the ML API service to be ready on startup
		this.serviceReady = apiManager.waitForService();
	}

	public async init(): Promise<void> {
		const isReady = await this.serviceReady;
		if (!isReady) {
			throw new WorkerError(
				new Error("ML API service failed to become ready"),
				"ml",
				"fn:init"
			);
		}
		logger.log("Akari API initialized successfully", "ml");
	}

	public async classifyVideo(
		title: string,
		description: string,
		tags: string,
		aid?: number
	): Promise<number> {
		try {
			// Ensure service is ready
			await this.serviceReady;

			const label = await apiManager.classifyVideo(title, description, tags, aid);
			return label;
		} catch (error) {
			logger.error(`Classification failed for aid ${aid}: ${error}`, "ml");
			throw new WorkerError(error as Error, "ml", "fn:classifyVideo");
		}
	}

	public async classifyVideosBatch(
		videos: Array<{ title: string; description: string; tags: string; aid?: number }>
	): Promise<Array<{ aid?: number; label: number; probabilities: number[] }>> {
		try {
			// Ensure service is ready
			await this.serviceReady;

			const results = await apiManager.classifyVideosBatch(videos);
			return results;
		} catch (error) {
			logger.error(`Batch classification failed: ${error}`, "ml");
			throw new WorkerError(error as Error, "ml", "fn:classifyVideosBatch");
		}
	}

	public async healthCheck(): Promise<boolean> {
		return await apiManager.healthCheck();
	}
}

// Create a singleton instance
const Akari = new AkariAPI();
export default Akari;
