import type { App } from "@backend/src";
import { treaty } from "@elysiajs/eden";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { ErrorPage } from "@/components/Error";
import { Layout } from "@/components/Layout";
import { Title } from "@/components/Title";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ControlBar } from "@/routes/labelling/ControlBar";
import { LabelInstructions } from "@/routes/labelling/LabelInstructions";
import { VideoInfo } from "./VideoInfo";

// @ts-expect-error anyway...
const app = treaty<App>(import.meta.env.VITE_API_URL!);

type VideosResponse = Awaited<ReturnType<Awaited<typeof app.videos.unlabelled>["get"]>>["data"];

const leftKeys = [
	"1",
	"2",
	"3",
	"4",
	"5",
	"Q",
	"W",
	"E",
	"R",
	"T",
	"A",
	"S",
	"D",
	"F",
	"G",
	"Z",
	"X",
	"C",
	"V",
	"B",
];
const rightKeys = [
	"6",
	"7",
	"8",
	"9",
	"0",
	"Y",
	"U",
	"I",
	"O",
	"P",
	"H",
	"J",
	"K",
	"L",
	";",
	"N",
	"M",
	",",
	".",
	"/",
];

export default function Home() {
	const [videos, setVideos] = useState<Exclude<VideosResponse, null>>([]);
	const [currentIndex, setCurrentIndex] = useState(0);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<any>(null);
	const [instructionsOpen, setInstructionsOpen] = useState(false);
	const [localLabel, setLocalLabel] = useState<(boolean | undefined)[]>([]);

	const fetchVideos = useCallback(async () => {
		if (videos.length >= 30) return;
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
			}
		} catch (err) {
			setError({ status: 500, value: { message: "网络错误" } });
		} finally {
			setLoading(false);
		}
	}, []);

	const loadMoreIfNeeded = useCallback(() => {
		if (videos.length - currentIndex <= 6) {
			fetchVideos();
		}
	}, [videos.length, currentIndex, fetchVideos]);

	const labelVideo = async (currentIndex: number, videoId: string, label: boolean) => {
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
					}
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
			setLocalLabel((prev) => {
				const newLabel = [...prev];
				newLabel[currentIndex] = undefined;
				return newLabel;
			});
		}
	};

	const handleLabel = async (label: boolean) => {
		const currentVideo = videos[currentIndex];
		if (!currentVideo) return;
		setLocalLabel((prev) => {
			const newLabel = [...prev];
			newLabel[currentIndex] = label;
			console.log(newLabel);
			return newLabel;
		});

		labelVideo(currentIndex, currentVideo.bvid!, label);
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

	useEffect(() => {
		const handleKeyDown = (e: KeyboardEvent) => {
			if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
				return;
			}

			const hasModifier = e.ctrlKey || e.altKey || e.shiftKey || e.metaKey;

			if (hasModifier) {
				return;
			}

			const key = e.key.toUpperCase();

			if (leftKeys.includes(key)) {
				handleLabel(false);
			} else if (rightKeys.includes(key)) {
				handleLabel(true);
			} else if (key === "ARROWLEFT") {
				navigateTo(currentIndex - 1);
			} else if (key === "ARROWRIGHT") {
				navigateTo(currentIndex + 1);
			}
		};

		window.addEventListener("keydown", handleKeyDown);
		return () => window.removeEventListener("keydown", handleKeyDown);
	}, [currentIndex, videos]);

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
		return <ErrorPage error={error} />;
	}

	const currentVideo = videos[currentIndex];
	const currentLabel = (() => {
		const l = localLabel[currentIndex];
		if (l === undefined) return "none";
		return l ? "true" : "false";
	})();

	return (
		<Layout>
			<Title title="视频打标工具" />

			{currentVideo ? (
				<>
					<LabelInstructions open={instructionsOpen} onOpenChange={setInstructionsOpen} />
					<span className="font-mono">
						Buffer health: {currentIndex}/{videos.length}, currentLabel: {currentLabel}
					</span>
					<VideoInfo video={currentVideo} />
					<ControlBar
						currentIndex={currentIndex}
						videosLength={videos.length}
						onPrevious={() => navigateTo(currentIndex - 1)}
						onNext={() => navigateTo(currentIndex + 1)}
						onLabel={handleLabel}
					/>
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
