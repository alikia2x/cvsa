import { AutoTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime";

// 模型路径和名称
const tokenizerModel = "alikia2x/jina-embedding-v3-m2v-1024";
const onnxClassifierPath = "./model/video_classifier_v3_11.onnx";
const onnxEmbeddingOriginalPath = "./model/model.onnx";
export const modelVersion = "3.11";

// 全局变量，用于存储模型和分词器
let sessionClassifier: ort.InferenceSession | null = null;
let sessionEmbedding: ort.InferenceSession | null = null;
let tokenizer: any | null = null;

// 初始化分词器和ONNX会话
async function initializeModels() {
	if (tokenizer && sessionClassifier && sessionEmbedding) {
		return; // 模型已加载，无需重复加载
	}

	try {
		const tokenizerConfig = { local_files_only: true };
		tokenizer = await AutoTokenizer.from_pretrained(tokenizerModel, tokenizerConfig);

		const [classifierSession, embeddingSession] = await Promise.all([
			ort.InferenceSession.create(onnxClassifierPath),
			ort.InferenceSession.create(onnxEmbeddingOriginalPath),
		]);

		sessionClassifier = classifierSession;
		sessionEmbedding = embeddingSession;
	} catch (error) {
		console.error("Error initializing models:", error);
		throw error; // 重新抛出错误，以便调用方处理
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

	// 构造输入参数
	const cumsum = (arr: number[]): number[] =>
		arr.reduce((acc: number[], num: number, i: number) => [...acc, num + (acc[i - 1] || 0)], []);

	const offsets: number[] = [0, ...cumsum(input_ids.slice(0, -1).map((x: string) => x.length))];
	const flattened_input_ids = input_ids.flat();

	// 准备ONNX输入
	const inputs = {
		input_ids: new ort.Tensor("int64", new BigInt64Array(flattened_input_ids.map(BigInt)), [
			flattened_input_ids.length,
		]),
		offsets: new ort.Tensor("int64", new BigInt64Array(offsets.map(BigInt)), [offsets.length]),
	};

	// 执行推理
	const { embeddings } = await session.run(inputs);
	return Array.from(embeddings.data as Float32Array);
}

// 分类推理函数
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

// 导出分类函数
export async function classifyVideo(
	title: string,
	description: string,
	tags: string,
	author_info: string,
): Promise<number[]> {
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
	return probabilities;
}
