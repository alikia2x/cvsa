import networkDelegate from "./delegate.ts";
import { MediaListInfoData, MediaListInfoResponse } from "net/bilibili.d.ts";
import logger from "log/logger.ts";

/*
 * Bulk fetch video metadata from bilibili API
 * @param {number[]} aids - The aid list to fetch
 * @returns {Promise<MediaListInfoData | number>} MediaListInfoData or the error code returned by bilibili API
 * @throws {NetSchedulerError} - The error will be thrown in following cases:
 * - No proxy is available currently: with error code `NO_PROXY_AVAILABLE`
 * - The native `fetch` function threw an error: with error code `FETCH_ERROR`
 * - The alicloud-fc threw an error: with error code `ALICLOUD_FC_ERROR`
 */
export async function bulkGetVideoStats(aids: number[]): Promise<MediaListInfoData | number> {
	let url = `https://api.bilibili.com/medialist/gateway/base/resource/infos?resources=`;
	for (const aid of aids) {
		url += `${aid}:2,`;
	}
	const data = await networkDelegate.request<MediaListInfoResponse>(url, "bulkSnapshot");
	const errMessage = `Error fetching metadata for aid list: ${aids.join(",")}:`;
	if (data.code !== 0) {
		logger.error(errMessage + data.code + "-" + data.message, "net", "fn:getVideoInfo");
		return data.code;
	}
	return data.data;
}
