import logger from "@core/log";
import { WorkerError } from "mq/schema";

interface ClassificationRequest {
    title: string;
    description: string;
    tags: string;
    aid?: number;
}

interface ClassificationResponse {
    label: number;
    probabilities: number[];
    aid?: number;
}

interface HealthResponse {
    status: string;
    models_loaded: boolean;
}

export class APIManager {
    private readonly baseUrl: string;
    private readonly timeout: number;

    constructor(baseUrl: string = "http://localhost:8544", timeout: number = 30000) {
        this.baseUrl = baseUrl;
        this.timeout = timeout;
    }

    public async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: AbortSignal.timeout(this.timeout),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data: HealthResponse = await response.json();
            return data.models_loaded;
        } catch (error) {
            logger.error(`Health check failed: ${error}`, "ml");
            return false;
        }
    }

    public async classifyVideo(
        title: string,
        description: string,
        tags: string,
        aid?: number
    ): Promise<number> {
        const request: ClassificationRequest = {
            title: title.trim() || "untitled",
            description: description.trim() || "N/A",
            tags: tags.trim() || "empty",
            aid: aid
        };

        try {
            const response = await fetch(`${this.baseUrl}/classify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
                signal: AbortSignal.timeout(this.timeout),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data: ClassificationResponse = await response.json();
            
            if (aid) {
                logger.log(
                    `Prediction result for aid: ${aid}: [${data.probabilities.map((p) => p.toFixed(5))}]`,
                    "ml"
                );
            }

            return data.label;
        } catch (error) {
            logger.error(`Classification failed for aid ${aid}: ${error}`, "ml");
            throw new WorkerError(error as Error, "ml", "fn:classifyVideo");
        }
    }

    public async classifyVideosBatch(
        requests: Array<{ title: string; description: string; tags: string; aid?: number }>
    ): Promise<Array<{ aid?: number; label: number; probabilities: number[] }>> {
        try {
            const response = await fetch(`${this.baseUrl}/classify_batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requests),
                signal: AbortSignal.timeout(this.timeout * 2), // Longer timeout for batch
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.results;
        } catch (error) {
            logger.error(`Batch classification failed: ${error}`, "ml");
            throw new WorkerError(error as Error, "ml", "fn:classifyVideosBatch");
        }
    }

    public async waitForService(timeoutMs: number = 60000): Promise<boolean> {
        const startTime = Date.now();
        const checkInterval = 2000; // Check every 2 seconds

        while (Date.now() - startTime < timeoutMs) {
            try {
                const isHealthy = await this.healthCheck();
                if (isHealthy) {
                    logger.log("ML API service is healthy", "ml");
                    return true;
                }
            } catch (error) {
                // Service not ready yet, continue waiting
            }

            await new Promise(resolve => setTimeout(resolve, checkInterval));
        }

        logger.error("ML API service did not become ready within timeout", "ml");
        return false;
    }
}

// Create a singleton instance
const apiManager = new APIManager();
export default apiManager;