import { AIManager } from "lib/ml/manager.ts";
import * as ort from "onnxruntime";
import logger from "lib/log/logger.ts";
import { WorkerError } from "lib/mq/schema.ts";

const modelPath = "./model/model.onnx";

class MantisProto extends AIManager {

	constructor() {
		super();
        this.models = {
            "predictor": modelPath,
        }
	}

    public override async init(): Promise<void> {
        await super.init();
    }

    
}

const Mantis = new MantisProto();
export default Mantis;
