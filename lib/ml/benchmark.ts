import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime";
import { softmax } from "lib/ml/filter_inference.ts";

// 配置参数
const sentenceTransformerModelName = "alikia2x/jina-embedding-v3-m2v-1024";
const onnxClassifierPath = "./model/video_classifier_v3_17.onnx";
const onnxEmbeddingPath = "./model/embedding_original.onnx";
const testDataPath = "./data/filter/test1.jsonl";

// 初始化会话
const [sessionClassifier, sessionEmbedding] = await Promise.all([
	ort.InferenceSession.create(onnxClassifierPath),
	ort.InferenceSession.create(onnxEmbeddingPath),
]);

let tokenizer: PreTrainedTokenizer;

// 初始化分词器
async function loadTokenizer() {
	const tokenizerConfig = { local_files_only: true };
	tokenizer = await AutoTokenizer.from_pretrained(sentenceTransformerModelName, tokenizerConfig);
}

// 新的嵌入生成函数（使用ONNX）
async function getONNXEmbeddings(texts: string[], session: ort.InferenceSession): Promise<number[]> {
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
	const inputTensor = new ort.Tensor(
		Float32Array.from(embeddings),
		[1, 3, 1024],
	);

	const { logits } = await sessionClassifier.run({ channel_features: inputTensor });
	return softmax(logits.data as Float32Array);
}

// 指标计算函数
function calculateMetrics(labels: number[], predictions: number[], elapsedTime: number): {
	accuracy: number;
	precision: number;
	recall: number;
	f1: number;
	"Class 0 Prec": number;
	speed: string;
} {
	// 输出label和prediction不一样的index列表
	const arr = [];
	for (let i = 0; i < labels.length; i++) {
		if (labels[i] !== predictions[i] && predictions[i] == 0) {
			arr.push([i + 1, labels[i], predictions[i]]);
		}
	}
	console.log(arr);
	// 初始化混淆矩阵
	const classCount = Math.max(...labels, ...predictions) + 1;
	const matrix = Array.from({ length: classCount }, () => Array.from({ length: classCount }, () => 0));

	// 填充矩阵
	labels.forEach((trueLabel, i) => {
		matrix[trueLabel][predictions[i]]++;
	});

	// 计算各指标
	let totalTP = 0, totalFP = 0, totalFN = 0;

	for (let c = 0; c < classCount; c++) {
		const TP = matrix[c][c];
		const FP = matrix.flatMap((row, i) => i === c ? [] : [row[c]]).reduce((a, b) => a + b, 0);
		const FN = matrix[c].filter((_, i) => i !== c).reduce((a, b) => a + b, 0);

		totalTP += TP;
		totalFP += FP;
		totalFN += FN;
	}

	const precision = totalTP / (totalTP + totalFP);
	const recall = totalTP / (totalTP + totalFN);
	const f1 = 2 * (precision * recall) / (precision + recall) || 0;

	// 计算Class 0 Precision
	const class0TP = matrix[0][0];
	const class0FP = matrix.flatMap((row, i) => i === 0 ? [] : [row[0]]).reduce((a, b) => a + b, 0);
	const class0Precision = class0TP / (class0TP + class0FP) || 0;

	return {
		accuracy: labels.filter((l, i) => l === predictions[i]).length / labels.length,
		precision,
		recall,
		f1,
		speed: `${(labels.length / (elapsedTime / 1000)).toFixed(1)} samples/sec`,
		"Class 0 Prec": class0Precision,
	};
}

// 改造后的评估函数
async function evaluateModel(session: ort.InferenceSession): Promise<{
	accuracy: number;
	precision: number;
	recall: number;
	f1: number;
	"Class 0 Prec": number;
}> {
	const data = await Deno.readTextFile(testDataPath);
	const samples = data.split("\n")
		.map((line) => {
			try {
				return JSON.parse(line);
			} catch {
				return null;
			}
		})
		.filter(Boolean);

	const allPredictions: number[] = [];
	const allLabels: number[] = [];

	const t = new Date().getTime();
	for (const sample of samples) {
		try {
			const embeddings = await getONNXEmbeddings([
				sample.title,
				sample.description,
				sample.tags.join(","),
			], session);

			const probabilities = await runClassification(embeddings);
			allPredictions.push(probabilities.indexOf(Math.max(...probabilities)));
			allLabels.push(sample.label);
		} catch (error) {
			console.error("Processing error:", error);
		}
	}
	const elapsed = new Date().getTime() - t;

	return calculateMetrics(allLabels, allPredictions, elapsed);
}

// 主函数
async function main() {
	await loadTokenizer();

	const metrics = await evaluateModel(sessionEmbedding);
	console.log("Model Metrics:");
	console.table(metrics);
}

await main();
