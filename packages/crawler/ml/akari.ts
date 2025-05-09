import { AIManager } from "ml/manager.ts";
import * as ort from "onnxruntime-node";
import logger from "@core/log/logger.ts";
import { WorkerError } from "mq/schema.ts";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import { AkariModelVersion } from "./const";

const tokenizerModel = "alikia2x/jina-embedding-v3-m2v-1024";
const onnxClassifierPath = `../../model/akari/${AkariModelVersion}.onnx`;
const onnxEmbeddingPath = "../../model/embedding/model.onnx";

class AkariProto extends AIManager {
	private tokenizer: PreTrainedTokenizer | null = null;
	private readonly modelVersion = AkariModelVersion;

	constructor() {
		super();
		this.models = {
			"classifier": onnxClassifierPath,
			"embedding": onnxEmbeddingPath,
		};
	}

	public override async init(): Promise<void> {
		await super.init();
		await this.initJinaTokenizer();
	}

	private tokenizerInitialized(): boolean {
		return this.tokenizer !== null;
	}

	private getTokenizer(): PreTrainedTokenizer {
		if (!this.tokenizerInitialized()) {
			throw new Error("Tokenizer is not initialized. Call init() first.");
		}
		return this.tokenizer!;
	}

	private async initJinaTokenizer(): Promise<void> {
		if (this.tokenizerInitialized()) {
			return;
		}
		try {
			this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerModel);
			logger.log("Tokenizer initialized", "ml");
		} catch (error) {
			throw new WorkerError(error as Error, "ml", "fn:initTokenizer");
		}
	}

	private async getJinaEmbeddings1024(texts: string[]): Promise<number[]> {
		const tokenizer = this.getTokenizer();
		const session = this.getModelSession("embedding");

		const { input_ids } = await tokenizer(texts, {
			add_special_tokens: false,
			return_tensor: false,
		});

		const cumsum = (arr: number[]): number[] =>
			arr.reduce((acc: number[], num: number, i: number) => [...acc, num + (acc[i - 1] || 0)], []);

		const offsets: number[] = [0, ...cumsum(input_ids.slice(0, -1).map((x: string) => x.length))];
		const flattened_input_ids = input_ids.flat();

		const inputs = {
			input_ids: new ort.Tensor("int64", new BigInt64Array(flattened_input_ids.map(BigInt)), [
				flattened_input_ids.length,
			]),
			offsets: new ort.Tensor("int64", new BigInt64Array(offsets.map(BigInt)), [offsets.length]),
		};

		const { embeddings } = await session.run(inputs);
		return Array.from(embeddings.data as Float32Array);
	}

	private async runClassification(embeddings: number[]): Promise<number[]> {
		const session = this.getModelSession("classifier");
		const inputTensor = new ort.Tensor(
			Float32Array.from(embeddings),
			[1, 3, 1024],
		);

		const { logits } = await session.run({ channel_features: inputTensor });
		return this.softmax(logits.data as Float32Array);
	}

	public async classifyVideo(title: string, description: string, tags: string, aid?: number): Promise<number> {
		const embeddings = await this.getJinaEmbeddings1024([
			title,
			description,
			tags,
		]);
		const probabilities = await this.runClassification(embeddings);
		if (aid) {
			logger.log(`Prediction result for aid: ${aid}: [${probabilities.map((p) => p.toFixed(5))}]`, "ml");
		}
		return probabilities.indexOf(Math.max(...probabilities));
	}

	public getModelVersion(): string {
		return this.modelVersion;
	}
}

const Akari = new AkariProto();
await Akari.init();
export default Akari;
