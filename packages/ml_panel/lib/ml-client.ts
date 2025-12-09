// ML Client for communicating with FastAPI service
import type { TrainingConfig, ExperimentResult } from './types';

export interface Hyperparameter {
  name: string;
  type: 'number' | 'boolean' | 'select';
  value: any;
  range?: [number, number];
  options?: string[];
  description?: string;
}

export interface TrainingRequest {
  experimentName: string;
  config: TrainingConfig;
  dataset: {
    aid: number[];
    embeddings: Record<string, number[]>;
    labels: Record<number, boolean>;
  };
}

export interface TrainingStatus {
  experimentId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  currentEpoch?: number;
  totalEpochs?: number;
  metrics?: Record<string, number>;
  error?: string;
}

export class MLClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  // Get available hyperparameters from the model
  async getHyperparameters(): Promise<Hyperparameter[]> {
    const response = await fetch(`${this.baseUrl}/hyperparameters`);
    if (!response.ok) {
      throw new Error(`Failed to get hyperparameters: ${response.statusText}`);
    }
    return (await response.json()) as Hyperparameter[];
  }

  // Start a training experiment
  async startTraining(request: TrainingRequest): Promise<{ experimentId: string }> {
    const response = await fetch(`${this.baseUrl}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to start training: ${response.statusText}`);
    }
    return (await response.json()) as { experimentId: string };
  }

  // Get training status
  async getTrainingStatus(experimentId: string): Promise<TrainingStatus> {
    const response = await fetch(`${this.baseUrl}/train/${experimentId}/status`);
    if (!response.ok) {
      throw new Error(`Failed to get training status: ${response.statusText}`);
    }
    return (await response.json()) as TrainingStatus;
  }

  // Get experiment results
  async getExperimentResult(experimentId: string): Promise<ExperimentResult> {
    const response = await fetch(`${this.baseUrl}/experiments/${experimentId}`);
    if (!response.ok) {
      throw new Error(`Failed to get experiment result: ${response.statusText}`);
    }
    return (await response.json()) as ExperimentResult;
  }

  // List all experiments
  async listExperiments(): Promise<ExperimentResult[]> {
    const response = await fetch(`${this.baseUrl}/experiments`);
    if (!response.ok) {
      throw new Error(`Failed to list experiments: ${response.statusText}`);
    }
    return (await response.json()) as ExperimentResult[];
  }

  // Generate embeddings using OpenAI-compatible API
  async generateEmbeddings(texts: string[], model: string): Promise<number[][]> {
    const response = await fetch(`${this.baseUrl}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ texts, model }),
    });

    if (!response.ok) {
      throw new Error(`Failed to generate embeddings: ${response.statusText}`);
    }
    return (await response.json()) as number[][];
  }
}