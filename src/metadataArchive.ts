import arg from "arg";
//import { getVideoDetails } from "@crawler/net/getVideoDetails";
import logger from "@core/log/logger";

const quit = (reason: string) => {
	logger.error(reason);
	process.exit();
};

const args = arg({
	"--aids": String // --port <number> or --port=<number>
});

const aidsFileName = args["--aids"];

if (!aidsFileName) {
	quit("Missing --aids <file_path>");
}

const aidsFile = Bun.file(aidsFileName!);
const fileExists = await aidsFile.exists();
if (!fileExists) {
	quit(`${aidsFile} does not exist.`);
}

const aidsText = await aidsFile.text();
const aids = aidsText
	.split("\n")
	.map((line) => parseInt(line))
	.filter((num) => !Number.isNaN(num));

logger.log(`Read ${aids.length} aids.`);
