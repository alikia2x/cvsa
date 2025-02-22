import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime";
import logger from "lib/log/logger.ts";
import { WorkerError } from "../mq/schema.ts";

const tokenizerModel = "alikia2x/jina-embedding-v3-m2v-1024";
const onnxClassifierPath = "./model/video_classifier_v3_11.onnx";
const onnxEmbeddingOriginalPath = "./model/model.onnx";
export const modelVersion = "3.11";

let sessionClassifier: ort.InferenceSession | null = null;
let sessionEmbedding: ort.InferenceSession | null = null;
let tokenizer: PreTrainedTokenizer | null = null;

export async function initializeModels() {
	if (tokenizer && sessionClassifier && sessionEmbedding) {
		return;
	}

	try {
		tokenizer = await AutoTokenizer.from_pretrained(tokenizerModel);

		const [classifierSession, embeddingSession] = await Promise.all([
			ort.InferenceSession.create(onnxClassifierPath),
			ort.InferenceSession.create(onnxEmbeddingOriginalPath),
		]);

		sessionClassifier = classifierSession;
		sessionEmbedding = embeddingSession;
		logger.log("Filter models initialized", "ml");
	} catch (error) {
		const e = new WorkerError(error as Error, "ml", "fn:initializeModels");
		throw e;
	}
}

function softmax(logits: Float32Array): number[] {
	const maxLogit = Math.max(...logits);
	const exponents = logits.map((logit) => Math.exp(logit - maxLogit));
	const sumOfExponents = exponents.reduce((sum, exp) => sum + exp, 0);
	return Array.from(exponents.map((exp) => exp / sumOfExponents));
}

async function getONNXEmbeddings(texts: string[], session: ort.InferenceSession): Promise<number[]> {
	if (!tokenizer) {
		throw new Error("Tokenizer is not initialized. Call initializeModels() first.");
	}
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


async function runClassification(embeddings: number[]): Promise<number[]> {
	if (!sessionClassifier) {
		throw new Error("Classifier session is not initialized. Call initializeModels() first.");
	}
	const inputTensor = new ort.Tensor(
		Float32Array.from(embeddings),
		[1, 4, 1024],
	);

	const { logits } = await sessionClassifier.run({ channel_features: inputTensor });
	return softmax(logits.data as Float32Array);
}

export async function classifyVideo(
	title: string,
	description: string,
	tags: string,
	author_info: string,
	aid: number
): Promise<number> {
	if (!sessionEmbedding) {
		throw new Error("Embedding session is not initialized. Call initializeModels() first.");
	}
	const embeddings = await getONNXEmbeddings([
		title,
		description,
		tags,
		author_info,
	], sessionEmbedding);
	const probabilities = await runClassification(embeddings);
	logger.log(`Prediction result for aid: ${aid}: [${probabilities.map((p) => p.toFixed(5))}]`, "ml")
	return probabilities.indexOf(Math.max(...probabilities));
}
