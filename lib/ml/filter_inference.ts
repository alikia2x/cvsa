import { AutoTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime";

// 配置参数
const sentenceTransformerModelName = "alikia2x/jina-embedding-v3-m2v-1024";
const onnxClassifierPath = "./model/video_classifier_v3_11.onnx";
const onnxEmbeddingOriginalPath = "./model/model.onnx";

// 初始化会话
const [sessionClassifier, sessionEmbedding] = await Promise.all([
    ort.InferenceSession.create(onnxClassifierPath),
    ort.InferenceSession.create(onnxEmbeddingOriginalPath),
]);


let tokenizer: any;

// 初始化分词器
async function loadTokenizer() {
    const tokenizerConfig = { local_files_only: true };
    tokenizer = await AutoTokenizer.from_pretrained(sentenceTransformerModelName, tokenizerConfig);
}

function softmax(logits: Float32Array): number[] {
	const maxLogit = Math.max(...logits);
	const exponents = logits.map((logit) => Math.exp(logit - maxLogit));
	const sumOfExponents = exponents.reduce((sum, exp) => sum + exp, 0);
	return Array.from(exponents.map((exp) => exp / sumOfExponents));
}

async function getONNXEmbeddings(texts: string[], session: ort.InferenceSession): Promise<number[]> {
    const { input_ids } = await tokenizer(texts, { 
        add_special_tokens: false, 
        return_tensor: false 
    });

    // 构造输入参数
    const cumsum = (arr: number[]): number[] =>
        arr.reduce((acc: number[], num: number, i: number) => [...acc, num + (acc[i - 1] || 0)], []);
    
    const offsets: number[] = [0, ...cumsum(input_ids.slice(0, -1).map((x: string) => x.length))];
    const flattened_input_ids = input_ids.flat();

    // 准备ONNX输入
    const inputs = {
        input_ids: new ort.Tensor("int64", new BigInt64Array(flattened_input_ids.map(BigInt)), [flattened_input_ids.length]),
        offsets: new ort.Tensor("int64", new BigInt64Array(offsets.map(BigInt)), [offsets.length])
    };

    // 执行推理
    const { embeddings } = await session.run(inputs);
    return Array.from(embeddings.data as Float32Array);
}

// 分类推理函数
async function runClassification(embeddings: number[]): Promise<number[]> {
    const inputTensor = new ort.Tensor(
        Float32Array.from(embeddings), 
        [1, 4, 1024]
    );
    
    const { logits } = await sessionClassifier.run({ channel_features: inputTensor });
    return softmax(logits.data as Float32Array);
}

async function processInputTexts(
	title: string,
	description: string,
	tags: string,
	author_info: string,
): Promise<number[]> {
	const embeddings = await getONNXEmbeddings([
		title,
		description,
		tags,
		author_info
	], sessionEmbedding);

	const probabilities = await runClassification(embeddings);
	return probabilities;
}

async function main() {
	await loadTokenizer();
	const titleText = `【洛天依&乐正绫&心华原创】归一【时之歌Project】`
    const descriptionText = " 《归一》Vocaloid ver\r\n出品：泛音堂 / 作词：冥凰 / 作曲：汤汤 / 编曲&amp;混音：iAn / 调教：花之祭P\r\n后期：向南 / 人设：Pora / 场景：A舍长 / PV：Sung Hsu（麻薯映画） / 海报：易玄玑 \r\n唱：乐正绫 &amp; 洛天依 &amp; 心华\r\n时之歌Project东国世界观歌曲《归一》双本家VC版\r\nMP3：http://5sing.kugou.com/yc/3006072.html \r\n伴奏：http://5sing.kugou.com/bz/2";
    const tagsText = '乐正绫,洛天依,心华,VOCALOID中文曲,时之歌,花之祭P';
    const authorInfoText = "时之歌Project: 欢迎光临时之歌~\r\n官博：http://weibo.com/songoftime\r\n官网：http://www.songoftime.com/";

	try {
		const probabilities = await processInputTexts(titleText, descriptionText, tagsText, authorInfoText);
		console.log("Class Probabilities:", probabilities);
		console.log(`Class 0 Probability: ${probabilities[0]}`);
		console.log(`Class 1 Probability: ${probabilities[1]}`);
		console.log(`Class 2 Probability: ${probabilities[2]}`);
        // Hold the session for 10s
		await new Promise((resolve) => setTimeout(resolve, 10000));
	} catch (error) {
		console.error("Error processing texts:", error);
	}
}

await main();
