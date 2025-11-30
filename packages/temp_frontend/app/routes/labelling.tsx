import { Layout } from "@/components/Layout";
import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { formatDateTime } from "@/components/SearchResults";
import { treaty } from "@elysiajs/eden";
import type { App } from "@backend/src";
import { Skeleton } from "@/components/ui/skeleton";
import { Error } from "@/components/Error";
import { Title } from "@/components/Title";
import { toast } from "sonner";
import { ChevronLeft, ChevronRight, Check, X } from "lucide-react";

// @ts-expect-error anyway...
const app = treaty<App>(import.meta.env.VITE_API_URL!);

type VideosResponse = Awaited<ReturnType<Awaited<typeof app.videos.unlabelled>["get"]>>["data"];

export default function Home() {
	const [videos, setVideos] = useState<Exclude<VideosResponse, null>>([]);
	const [currentIndex, setCurrentIndex] = useState(0);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<any>(null);
	const [hasMore, setHasMore] = useState(true);

	const fetchVideos = useCallback(async () => {
		try {
			setLoading(true);
			const { data, error } = await app.videos.unlabelled.get({
				headers: {
					Authorization: `Bearer ${localStorage.getItem("sessionID") || ""}`,
				},
			});

			if (error) {
				setError(error);
				return;
			}

			if (data && data.length > 0) {
				setVideos((prev) => [...prev, ...data]);
				setHasMore(data.length === 20);
			} else {
				setHasMore(false);
			}
		} catch (err) {
			setError({ status: 500, value: { message: "网络错误" } });
		} finally {
			setLoading(false);
		}
	}, []);

	const loadMoreIfNeeded = useCallback(() => {
		if (hasMore && videos.length - currentIndex <= 6) {
			fetchVideos();
		}
	}, [hasMore, videos.length, currentIndex, fetchVideos]);

	const labelVideo = async (videoId: string, label: boolean) => {
		const videoKey = `${videoId}-${label}`;

		const maxRetries = 5;
		let retries = 0;

		const attemptLabel = async (): Promise<boolean> => {
			try {
				const { error } = await app.video({ id: videoId }).label.post(
					{ label },
					{
						headers: {
							Authorization: `Bearer ${localStorage.getItem("sessionID") || ""}`,
						},
					},
				);

				if (error) {
					throw error;
				}

				toast.success(`已标记视频 ${label ? "是" : "否"}`);
				return true;
			} catch (err) {
				retries++;
				if (retries < maxRetries) {
					await new Promise((resolve) => setTimeout(resolve, 1000 * retries));
					return attemptLabel();
				}
				return false;
			}
		};

		const success = await attemptLabel();

		if (!success) {
			toast.error(`标记失败，请稍后重试`);
		}
	};

	const handleLabel = async (label: boolean) => {
		const currentVideo = videos[currentIndex];
		if (!currentVideo) return;

		labelVideo(currentVideo.bvid!, label);
		if (currentIndex < videos.length - 1) {
			setCurrentIndex((prev) => prev + 1);
			loadMoreIfNeeded();
		} else {
			fetchVideos();
			if (videos.length > currentIndex + 1) {
				setCurrentIndex((prev) => prev + 1);
			}
		}
	};

	const navigateTo = (index: number) => {
		if (index >= 0 && index < videos.length) {
			setCurrentIndex(index);
			loadMoreIfNeeded();
		}
	};

	useEffect(() => {
		fetchVideos();
	}, [fetchVideos]);

	useEffect(() => {
		loadMoreIfNeeded();
	}, [currentIndex, loadMoreIfNeeded]);

	if (loading && videos.length === 0) {
		return (
			<Layout>
				<Title title="视频打标工具" />
				<div className="space-y-6">
					<Skeleton className="mt-6 w-full aspect-video rounded-lg" />
					<div className="mt-6 flex justify-between items-baseline">
						<Skeleton className="w-60 h-10 rounded-sm" />
						<Skeleton className="w-25 h-10 rounded-sm" />
					</div>
					<Skeleton className="w-full h-20 rounded-lg" />
				</div>
			</Layout>
		);
	}

	if (error && videos.length === 0) {
		return <Error error={error} />;
	}

	const currentVideo = videos[currentIndex];

	return (
		<Layout>
			<Title title="视频打标工具" />

			{currentVideo ? (
				<>
					<div className="mb-24">
						<p className="mt-4 mb-3">
							该视频是否包含一首<b>中V歌曲</b>？
							<Label className="text-secondary-foreground mt-1 leading-5">
								中V歌曲意味着它是由中文虚拟歌姬演唱，或歌词中包含中文。歌曲可以是原创，也可以是非原创（如翻唱、翻调等）。
							</Label>
						</p>
						<div className="flex flex-col sm:flex-row sm:gap-4">
							{currentVideo.cover_url && (
								<a
									href={`https://www.bilibili.com/video/${currentVideo.bvid}`}
									target="_blank"
									rel="noopener noreferrer"
									className="min-w-full sm:min-w-60 md:min-w-80 max-w-full
									sm:max-w-60 md:max-w-80 aspect-video"
								>
									<img
										src={currentVideo.cover_url}
										referrerPolicy="no-referrer"
										className="w-full object-cover rounded-lg"
										alt="Video cover"
									/>
								</a>
							)}
							<div>
								<div className="max-sm:mt-6 flex items-center gap-2">
									<h1 className="text-2xl font-medium">
										<a
											href={`https://www.bilibili.com/video/${currentVideo.bvid}`}
											target="_blank"
											rel="noopener noreferrer"
											className="hover:underline"
										>
											{currentVideo.title ? currentVideo.title : "未知视频标题"}
										</a>
									</h1>
								</div>

								<div className="flex justify-between mt-3">
									<div>
										<p>
											<span>{currentVideo.bvid}</span> · <span>av{currentVideo.aid}</span>
										</p>
										<p>
											<span>发布于 {formatDateTime(new Date(currentVideo.published_at!))}</span><br/>
											<span>播放：{(currentVideo.views ?? 0).toLocaleString()}</span>
										</p>
										<p>
											UP主：
											<a
												className="underline"
												href={`https://space.bilibili.com/${currentVideo.uid}`}
												target="_blank"
												rel="noopener noreferrer"
											>
												{currentVideo.username}
											</a>
										</p>
										<p>
											<span>
												<b>标签</b>
												<br />
												{currentVideo.tags?.replaceAll(",","，")}
											</span>
										</p>
									</div>
								</div>
							</div>
						</div>

						<div className="mt-6">
							<h3 className="font-medium text-lg mb-2">简介</h3>
							<pre className="max-w-full wrap-anywhere break-all text-on-surface-variant text-sm md:text-base whitespace-pre-wrap dark:text-dark-on-surface-variant font-zh">
								{currentVideo.description || "暂无简介"}
							</pre>
						</div>
					</div>

					<div className="fixed bottom-0 left-0 right-0 bg-background border-t p-4 shadow-lg">
						<div className="max-w-4xl mx-auto flex items-center justify-between gap-4">
							<Button
								variant="outline"
								onClick={() => navigateTo(currentIndex - 1)}
								disabled={currentIndex === 0}
								className="flex items-center gap-2"
							>
								<ChevronLeft className="h-4 w-4" />
								上一个
							</Button>

							<div className="flex gap-4">
								<Button
									variant="destructive"
									onClick={() => handleLabel(false)}
									className="flex items-center gap-2"
								>
									<X className="h-4 w-4" />否
								</Button>
								<Button
									variant="default"
									onClick={() => handleLabel(true)}
									className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
								>
									<Check className="h-4 w-4" />是
								</Button>
							</div>

							<Button
								variant="outline"
								onClick={() => navigateTo(currentIndex + 1)}
								disabled={currentIndex === videos.length - 1 && !hasMore}
								className="flex items-center gap-2"
							>
								下一个
								<ChevronRight className="h-4 w-4" />
							</Button>
						</div>
					</div>
				</>
			) : (
				<div className="text-center py-12">
					<p className="text-lg">没有更多视频需要打标</p>
					<Button onClick={fetchVideos} className="mt-4" disabled={loading}>
						{loading ? "加载中..." : "重新加载"}
					</Button>
				</div>
			)}
		</Layout>
	);
}
