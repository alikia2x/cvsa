import { format } from "date-fns";
import { zhCN } from "date-fns/locale";
import { getAllSnapshots } from "@/lib/db/snapshots/getAllSnapshots";
import { getAidFromBV } from "@/lib/db/bilibili_metadata/getAidFromBV";
import { getVideoMetadata } from "@/lib/db/bilibili_metadata/getVideoMetadata";
import { aidExists as idExists } from "@/lib/db/bilibili_metadata/aidExists";
import { notFound } from "next/navigation";
import { BiliVideoMetadataType, VideoSnapshotType } from "@cvsa/core";
import { Metadata } from "next";

const MetadataRow = ({ title, desc }: { title: string; desc: string | number | undefined | null }) => {
	if (!desc) return <></>;
	return (
		<tr>
			<td className="max-w-14 min-w-14 md:max-w-24 md:min-w-24 border dark:border-zinc-500 px-2 md:px-3 py-2 font-semibold">
				{title}
			</td>
			<td className="break-all max-w-[calc(100vw-4.5rem)] border dark:border-zinc-500 px-4 py-2">{desc}</td>
		</tr>
	);
};

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
	const backendURL = process.env.BACKEND_URL;
	const { id } = await params;
	const res = await fetch(`${backendURL}/video/${id}/info`);
	if (!res.ok) {
		return {
			title: "页面未找到 - 中 V 档案馆"
		};
	}
	const data = await res.json();
	return {
		title: `${data.title} - 歌曲信息 - 中 V 档案馆`
	};
}

export default async function VideoInfoPage({ params }: { params: Promise<{ id: string }> }) {
	const { id } = await params;
	let videoInfo: BiliVideoMetadataType | null = null;
	let snapshots: VideoSnapshotType[] = [];

	async function getVideoAid(videoId: string | string[] | undefined) {
		if (!videoId) return null;
		const videoIdStr = Array.isArray(videoId) ? videoId[0] : videoId;
		if (videoIdStr?.startsWith("av")) {
			return parseInt(videoIdStr.slice(2));
		} else if (videoIdStr?.startsWith("BV")) {
			return getAidFromBV(videoIdStr);
		}
		return parseInt(videoIdStr);
	}

	const aid = await getVideoAid(id);

	if (!aid) {
		return notFound();
	}

	const exists = await idExists(aid);

	if (!exists) {
		return notFound();
	}

	try {
		const videoData = await getVideoMetadata(aid);
		const snapshotsData = await getAllSnapshots(aid);
		videoInfo = videoData;
		if (snapshotsData) {
			snapshots = snapshotsData;
		}
	} catch (e) {
		console.error(e);
	}

	if (!videoInfo) {
		return notFound();
	}

	return (
		<main className="flex flex-col items-center min-h-screen gap-8 mt-10 md:mt-6 relative z-0 overflow-x-auto pb-8">
			<div className="w-full lg:max-w-4xl lg:mx-auto lg:p-6">
				<h1 className="text-2xl font-medium ml-2 mb-4">
					视频信息:{" "}
					<a href={`https://www.bilibili.com/video/av${videoInfo.aid}`} className="underline">
						av{videoInfo.aid}
					</a>
				</h1>

				<div className="mb-6">
					<h2 className="px-2 mb-2 text-xl font-medium">基本信息</h2>
					<div className="overflow-x-auto max-w-full px-2">
						<table className="table-fixed">
							<tbody>
								<MetadataRow title="ID" desc={videoInfo.id} />
								<MetadataRow title="av 号" desc={videoInfo.aid} />
								<MetadataRow title="BV 号" desc={videoInfo.bvid} />
								<MetadataRow title="标题" desc={videoInfo.title} />
								<MetadataRow title="描述" desc={videoInfo.description} />
								<MetadataRow title="UID" desc={videoInfo.uid} />
								<MetadataRow title="标签" desc={videoInfo.tags} />
								<MetadataRow
									title="发布时间"
									desc={
										videoInfo.published_at
											? format(new Date(videoInfo.published_at), "yyyy-MM-dd HH:mm:ss", {
													locale: zhCN
												})
											: null
									}
								/>
								<MetadataRow title="时长 (秒)" desc={videoInfo.duration} />
								<MetadataRow
									title="创建时间"
									desc={format(new Date(videoInfo.created_at), "yyyy-MM-dd HH:mm:ss", {
										locale: zhCN
									})}
								/>
								<MetadataRow title="封面" desc={videoInfo?.cover_url} />
							</tbody>
						</table>
					</div>
				</div>

				<div>
					<h2 className="px-2 mb-2 text-xl font-medium">播放量历史数据</h2>
					{snapshots && snapshots.length > 0 ? (
						<div className="overflow-x-auto px-2">
							<table className="table-auto w-full">
								<thead>
									<tr>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">创建时间</th>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">观看</th>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">硬币</th>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">点赞</th>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">收藏</th>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">分享</th>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">弹幕</th>
										<th className="border dark:border-zinc-500 px-4 py-2 font-medium">评论</th>
									</tr>
								</thead>
								<tbody>
									{snapshots.map((snapshot) => (
										<tr key={snapshot.created_at}>
											<td className="border dark:border-zinc-500 px-4 py-2">
												{format(new Date(snapshot.created_at), "yyyy-MM-dd HH:mm:ss", {
													locale: zhCN
												})}
											</td>
											<td className="border dark:border-zinc-500 px-4 py-2">{snapshot.views}</td>
											<td className="border dark:border-zinc-500 px-4 py-2">{snapshot.coins}</td>
											<td className="border dark:border-zinc-500 px-4 py-2">{snapshot.likes}</td>
											<td className="border dark:border-zinc-500 px-4 py-2">
												{snapshot.favorites}
											</td>
											<td className="border dark:border-zinc-500 px-4 py-2">{snapshot.shares}</td>
											<td className="border dark:border-zinc-500 px-4 py-2">
												{snapshot.danmakus}
											</td>
											<td className="border dark:border-zinc-500 px-4 py-2">
												{snapshot.replies}
											</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					) : (
						<p>暂无历史数据。</p>
					)}
				</div>
			</div>
		</main>
	);
}
