import type { App } from "@backend/src";
import { treaty } from "@elysiajs/eden";
import { useCallback, useEffect, useState } from "react";
import { ErrorPage } from "@/components/Error";
import { Layout } from "@/components/Layout";
import { formatDateTime } from "@/components/SearchResults";
import { Title } from "@/components/Title";
import { Skeleton } from "@/components/ui/skeleton";
import type { Route } from "./+types/index";

const app = treaty<App>(import.meta.env.VITE_API_URL || "");

type VideoInfo = Awaited<ReturnType<ReturnType<typeof app.video>["info"]["get"]>>["data"];
type VideoInfoError = Awaited<ReturnType<ReturnType<typeof app.video>["info"]["get"]>>["error"];

// noinspection JSUnusedGlobalSymbols
export async function clientLoader({ params }: Route.LoaderArgs) {
	return { id: params.id };
}

const StatRow = ({ title, description }: { title: string; description?: number }) => {
	return (
		<div className="flex justify-between w-36">
			<span>{title}</span>
			<span>{description?.toLocaleString() ?? "N/A"}</span>
		</div>
	);
};

export default function VideoInfo({ loaderData }: Route.ComponentProps) {
	const [videoInfo, setData] = useState<VideoInfo | null>(null);
	const [error, setError] = useState<VideoInfoError | null>(null);

	const getInfo = useCallback(async () => {
		const { data, error } = await app.video({ id: loaderData.id }).info.get();
		if (error) {
			console.log(error);
			setError(error);
			return;
		}
		setData(data);
	}, [loaderData.id]);

	useEffect(() => {
		getInfo().then(()=>{});
	}, [getInfo]);

	if (!videoInfo && !error) {
		return (
			<Layout>
				<Title title="加载中" />
				<Skeleton className="mt-6 w-full aspect-video rounded-lg" />
				<div className="mt-6 flex justify-between items-baseline">
					<Skeleton className="w-60 h-10 rounded-sm" />
					<Skeleton className="w-25 h-10 rounded-sm" />
				</div>
			</Layout>
		);
	}

	if (error) {
		return <ErrorPage error={error} />;
	}

	return (
		<Layout>
			<Title title={videoInfo?.title ? `${videoInfo?.title} - 视频信息` : "视频信息"} />
			{videoInfo?.pic && (
				<img
					src={videoInfo?.pic}
					referrerPolicy="no-referrer"
					className="w-full aspect-video object-cover rounded-lg mt-6"
					alt="Video cover"
				/>
			)}
			<div className="mt-6 flex items-center gap-2">
				<h1 className="text-2xl font-medium">
					<a href={`https://www.bilibili.com/video/${videoInfo?.bvid}`}>
						{videoInfo?.title ? videoInfo?.title : "未知视频标题"}
					</a>
				</h1>
			</div>
			<div className="flex justify-between mt-3">
				<div>
					<p>
						<span>{videoInfo?.bvid}</span> · <span>av{videoInfo?.aid}</span>
					</p>
					{videoInfo?.pubdate && <p>
						<span>发布于 {formatDateTime(new Date(videoInfo?.pubdate * 1000))}</span>
					</p>}
					<p>
						<span>播放：{(videoInfo?.stat?.view ?? 0).toLocaleString()}</span> ·{" "}
						<span>弹幕：{(videoInfo?.stat?.danmaku ?? 0).toLocaleString()}</span>
					</p>
					<p>
						<span>
							分区: {videoInfo?.tname}, tid{videoInfo?.tid}
						</span>
						{videoInfo?.tname_v2 && (
							<span>
								{" "}
								· v2: {videoInfo?.tname_v2}, tid{videoInfo?.tid_v2}
							</span>
						)}
					</p>
					{videoInfo?.owner && (
						<p>
							UP主：
							<a
								className="underline"
								href={`https://space.bilibili.com/${videoInfo?.owner.mid}`}
							>
								{videoInfo?.owner.name}
							</a>
						</p>
					)}
				</div>
			</div>

			<div className="mt-6">
				<h3 className="font-medium text-lg mb-2">简介</h3>
				<pre className="max-w-full wrap-anywhere break-all text-on-surface-variant text-sm md:text-base whitespace-pre-wrap dark:text-dark-on-surface-variant font-zh">
					{videoInfo?.desc || "暂无简介"}
				</pre>
			</div>

			<div className="mb-6 mt-6 stat-num">
				<h2 className="mb-2 text-xl font-medium">统计数据</h2>
				<div className="flex flex-col gap-1">
					<StatRow title="播放" description={videoInfo?.stat?.view} />
					<StatRow title="点赞" description={videoInfo?.stat?.like} />
					<StatRow title="收藏" description={videoInfo?.stat?.favorite} />
					<StatRow title="硬币" description={videoInfo?.stat?.coin} />
					<StatRow title="评论" description={videoInfo?.stat?.reply} />
					<StatRow title="弹幕" description={videoInfo?.stat?.danmaku} />
					<StatRow title="分享" description={videoInfo?.stat?.share} />
				</div>
			</div>
		</Layout>
	);
}
