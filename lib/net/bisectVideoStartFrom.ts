import { getLatestVideos } from "lib/net/getLatestVideos.ts";
import { AllDataType } from "lib/db/schema.d.ts";

export async function getVideoPositionInNewList(timestamp: number): Promise<number | null | AllDataType[]> {
	const virtualPageSize = 50;

	let lowPage = 1;
	let highPage = 1;
	let foundUpper = false;
	while (true) {
		const ps = highPage < 2 ? 50 : 1
		const pn = highPage < 2 ? 1 : highPage * virtualPageSize;
		const fetchTags = highPage < 2 ? true : false;
		const videos = await getLatestVideos(pn, ps, 250, fetchTags);
		if (!videos || videos.length === 0) {
			break;
		}
		const lastVideo = videos[videos.length - 1];
		if (!lastVideo || !lastVideo.published_at) {
			break;
		}
		const lastTime = Date.parse(lastVideo.published_at);
		if (lastTime <= timestamp && highPage == 1) {
			return videos;
		}
		else if (lastTime <= timestamp) {
			foundUpper = true;
			break;
		} else {
			lowPage = highPage;
			highPage *= 2;
		}
	}

	if (!foundUpper) {
		return null;
	}

	let boundaryPage = highPage;
	let lo = lowPage;
	let hi = highPage;
	while (lo <= hi) {
		const mid = Math.floor((lo + hi) / 2);
		const videos = await getLatestVideos(mid * virtualPageSize, 1, 250, false);
		if (!videos) {
			return null;
		}
		if (videos.length === 0) {
			hi = mid - 1;
			continue;
		}
		const lastVideo = videos[videos.length - 1];
		if (!lastVideo || !lastVideo.published_at) {
			hi = mid - 1;
			continue;
		}
		const lastTime = Date.parse(lastVideo.published_at);
		if (lastTime > timestamp) {
			lo = mid + 1;
		} else {
			boundaryPage = mid;
			hi = mid - 1;
		}
	}

	const boundaryVideos = await getLatestVideos(boundaryPage, virtualPageSize, 250, false);
	let indexInPage = 0;
	if (boundaryVideos && boundaryVideos.length > 0) {
		for (let i = 0; i < boundaryVideos.length; i++) {
			const video = boundaryVideos[i];
			if (!video.published_at) {
				continue;
			}
			const videoTime = Date.parse(video.published_at);
			if (videoTime > timestamp) {
				indexInPage++;
			} else {
				break;
			}
		}
	}

	const count = (boundaryPage - 1) * virtualPageSize + indexInPage;

	const safetyMargin = 5;

	return count + safetyMargin;
}
