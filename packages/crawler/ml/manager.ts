import * as ort from "onnxruntime-node";
import logger from "@core/log";
import { WorkerError } from "mq/schema";

export class AIManager {
	public sessions: { [key: string]: ort.InferenceSession } = {};
	public models: { [key: string]: string } = {};

	constructor() {}

	public async init() {
		const modelKeys = Object.keys(this.models);
		for (const key of modelKeys) {
			try {
				this.sessions[key] = await ort.InferenceSession.create(this.models[key]);
				logger.log(`Model ${key} initialized`, "ml");
			} catch (error) {
				throw new WorkerError(error as Error, "ml", "fn:init");
			}
		}
	}

	public getModelSession(key: string): ort.InferenceSession {
		if (this.sessions[key] === undefined) {
			throw new WorkerError(
				new Error(`Model ${key} not found / not initialized.`),
				"ml",
				"fn:getModelSession"
			);
		}
		return this.sessions[key];
	}

	public softmax(logits: Float32Array): number[] {
		const maxLogit = Math.max(...logits);
		const exponents = logits.map((logit) => Math.exp(logit - maxLogit));
		const sumOfExponents = exponents.reduce((sum, exp) => sum + exp, 0);
		return Array.from(exponents.map((exp) => exp / sumOfExponents));
	}
}
