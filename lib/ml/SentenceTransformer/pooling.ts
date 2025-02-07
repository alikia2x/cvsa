import { Tensor } from "@huggingface/transformers";
//@ts-ignore
import { Callable } from "@huggingface/transformers/src/utils/core.js"; // Keep as is for now, might need adjustment

export interface PoolingConfig {
  word_embedding_dimension: number;
  pooling_mode_cls_token: boolean;
  pooling_mode_mean_tokens: boolean;
  pooling_mode_max_tokens: boolean;
  pooling_mode_mean_sqrt_len_tokens: boolean;
}

export interface PoolingInput {
  token_embeddings: Tensor;
  attention_mask: Tensor;
}

export interface PoolingOutput {
  sentence_embedding: Tensor;
}

export class Pooling extends Callable {
  constructor(private readonly config: PoolingConfig) {
    super();
  }

  // async _call(inputs: any) { // Keep if pooling functionality is needed
  //   return this.forward(inputs);
  // }

  // async forward(inputs: PoolingInput): PoolingOutput { // Keep if pooling functionality is needed

  // }
}