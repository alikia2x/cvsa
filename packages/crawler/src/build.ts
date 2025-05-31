import Bun from "bun";

await Bun.build({
	entrypoints: ["./src/filterWorker.ts"],
	outdir: "./build",
	target: "node"
});

const file = Bun.file("./build/filterWorker.js");
const code = await file.text();

const modifiedCode = code.replaceAll("../bin/napi-v3/", "../../../node_modules/onnxruntime-node/bin/napi-v3/");

await Bun.write("./build/filterWorker.js", modifiedCode);
