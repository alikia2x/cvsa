import z from "zod";

const XOR_CODE = 23442827791579n;
const MASK_CODE = 2251799813685247n;
const MAX_AID = 1n << 51n;
const BASE = 58n;

const data = "FcwAPNKTMug3GV5Lj7EJnHpWsx4tb8haYeviqBz6rkCy12mUSDQX9RdoZf";

export function av2bv(aid: number) {
	const bytes = ["B", "V", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0"];
	let bvIndex = bytes.length - 1;
	let tmp = (MAX_AID | BigInt(aid)) ^ XOR_CODE;
	while (tmp > 0) {
		bytes[bvIndex] = data[Number(tmp % BigInt(BASE))];
		tmp = tmp / BASE;
		bvIndex -= 1;
	}
	[bytes[3], bytes[9]] = [bytes[9], bytes[3]];
	[bytes[4], bytes[7]] = [bytes[7], bytes[4]];
	return bytes.join("") as `BV1${string}`;
}

export function bv2av(bvid: `BV1${string}`) {
	const bvidArr = Array.from<string>(bvid);
	[bvidArr[3], bvidArr[9]] = [bvidArr[9], bvidArr[3]];
	[bvidArr[4], bvidArr[7]] = [bvidArr[7], bvidArr[4]];
	bvidArr.splice(0, 3);
	const tmp = bvidArr.reduce((pre, bvidChar) => pre * BASE + BigInt(data.indexOf(bvidChar)), 0n);
	return Number((tmp & MASK_CODE) ^ XOR_CODE);
}

const BV_REGEX = /^BV1[1-9A-HJ-NP-Za-km-z]{9}$/;
const bvSchema = z.string().regex(BV_REGEX);
const AV_REGEX = /^av[0-9]+$/;
const avSchema = z.string().regex(AV_REGEX);

export function detectBiliID(id: string) {
	if (bvSchema.safeParse(id).success) {
		return {
			type: "bv" as const,
			id: id as `BV1${string}`
		};
	} else if (avSchema.safeParse(id).success) {
		return {
			type: "av" as const,
			id: id as `av${string}`
		};
	}
	return null;
}

export function biliIDToAID(id: string) {
	const detected = detectBiliID(id);
	if (!detected) {
		return null;
	}
	if (detected.type === "bv") {
		return bv2av(detected.id);
	} else {
		return Number.parseInt(detected.id.slice(2));
	}
}
