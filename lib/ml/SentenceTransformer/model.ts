// lib/ml/sentence_transformer_model.ts
import { AutoModel, AutoTokenizer, PretrainedOptions } from "@huggingface/transformers";

export class SentenceTransformer {
	constructor(
		private readonly tokenizer: AutoTokenizer,
		private readonly model: AutoModel,
	) {}

	static async from_pretrained(
		modelName: string,
		options?: PretrainedOptions,
	): Promise<SentenceTransformer> {
		if (!options) {
			options = {
				progress_callback: undefined,
				cache_dir: undefined,
				local_files_only: false,
				revision: "main",
			};
		}
		const tokenizer = await AutoTokenizer.from_pretrained(modelName, options);
		const model = await AutoModel.from_pretrained(modelName, options);

		return new SentenceTransformer(tokenizer, model);
	}

	async encode(sentences: string[]): Promise<any> { // Changed return type to 'any' for now to match console.log output
		//@ts-ignore
		const modelInputs = await this.tokenizer(sentences, {
			padding: true,
			truncation: true,
		});

		//@ts-ignore
		const outputs = await this.model(modelInputs);

		return outputs;
	}
}
