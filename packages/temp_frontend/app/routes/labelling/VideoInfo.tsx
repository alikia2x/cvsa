import { formatDateTime } from "@/components/SearchResults";
import { treaty } from "@elysiajs/eden";
import type { App } from "@backend/src";

// @ts-expect-error anyway...
const app = treaty<App>(import.meta.env.VITE_API_URL!);

type VideosResponse = Awaited<ReturnType<Awaited<typeof app.videos.unlabelled>["get"]>>["data"];


interface VideoInfoProps {
	video: Exclude<VideosResponse, null>[number];
}

export function VideoInfo({ video }: VideoInfoProps) {
	const formatDuration = (duration: number) => {
		return `${Math.floor(duration / 60)}:${(duration % 60).toString().padStart(2, "0")}`;
	};

	return (
		<div className="mb-24">
			<div className="flex flex-col sm:flex-row sm:gap-4">
				{video.cover_url && (
					<a
						href={`https://www.bilibili.com/video/${video.bvid}`}
						target="_blank"
						rel="noopener noreferrer"
						className="min-w-full sm:min-w-60 md:min-w-80 max-w-full
            sm:max-w-60 md:max-w-80 aspect-video"
					>
						<img
							src={video.cover_url}
							referrerPolicy="no-referrer"
							className="w-full h-full object-cover rounded-lg"
							alt="Video cover"
						/>
					</a>
				)}
				<div>
					<div className="max-sm:mt-6 flex items-center gap-2">
						<h1 className="text-2xl font-medium">
							<a
								href={`https://www.bilibili.com/video/${video.bvid}`}
								target="_blank"
								rel="noopener noreferrer"
								className="hover:underline"
							>
								{video.title ? video.title : "未知视频标题"}
							</a>
						</h1>
					</div>

					<div className="flex justify-between mt-3">
						<div>
							<p>
								<span>{video.bvid}</span> · <span>av{video.aid}</span>
							</p>
							<p>
								<span>发布于 {formatDateTime(new Date(video.published_at!))}</span>
								<br />
								<span>播放：{(video.views ?? 0).toLocaleString()}</span>
								&nbsp;&nbsp;
								<span>时长：{formatDuration(video.duration || 0)}</span>
							</p>
							<p>
								UP主：
								<a
									className="underline"
									href={`https://space.bilibili.com/${video.uid}`}
									target="_blank"
									rel="noopener noreferrer"
								>
									{video.username}
								</a>
							</p>
							<p>
								<span>
									<b>标签</b>
									<br />
									{video.tags?.replaceAll(",", "，")}
								</span>
							</p>
						</div>
					</div>
				</div>
			</div>

			<div className="mt-6">
				<h3 className="font-medium text-lg mb-2">简介</h3>
				<pre className="max-w-full wrap-anywhere break-all text-on-surface-variant text-sm md:text-base whitespace-pre-wrap dark:text-dark-on-surface-variant font-zh">
					{video.description || "暂无简介"}
				</pre>
			</div>
		</div>
	);
}
