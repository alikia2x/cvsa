import { getLatestVideos } from "lib/net/getLatestVideos.ts";
import { SECOND } from "$std/datetime/constants.ts";
import { VideoListVideo } from "lib/net/bilibili.d.ts";

export async function getVideoPositionInNewList(timestamp: number): Promise<number | null | VideoListVideo[]> {
	const virtualPageSize = 50;

	let lowPage = 1;
	let highPage = 1;
	let foundUpper = false;
	while (true) {
		const ps = highPage < 2 ? 50 : 1
		const pn = highPage < 2 ? 1 : highPage * virtualPageSize;
		const videos = await getLatestVideos(pn, ps);
		if (!videos || videos.length === 0) {
			break;
		}
		const lastVideo = videos[videos.length - 1];
		if (!lastVideo || !lastVideo.pubdate) {
			break;
		}
		const lastTime = lastVideo.pubdate * SECOND
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
		const videos = await getLatestVideos(mid * virtualPageSize, 1);
		if (!videos) {
			return null;
		}
		if (videos.length === 0) {
			hi = mid - 1;
			continue;
		}
		const lastVideo = videos[videos.length - 1];
		if (!lastVideo || !lastVideo.pubdate) {
			hi = mid - 1;
			continue;
		}
		const lastTime = lastVideo.pubdate * SECOND
		if (lastTime > timestamp) {
			lo = mid + 1;
		} else {
			boundaryPage = mid;
			hi = mid - 1;
		}
	}

	const boundaryVideos = await getLatestVideos(boundaryPage, virtualPageSize);
	let indexInPage = 0;
	if (boundaryVideos && boundaryVideos.length > 0) {
		for (let i = 0; i < boundaryVideos.length; i++) {
			const video = boundaryVideos[i];
			if (!video.pubdate) {
				continue;
			}
			const videoTime = video.pubdate * SECOND
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
