import type { Route } from "./+types/info";
import { treaty } from "@elysiajs/eden";
import type { App } from "@elysia/src";
import { useEffect, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { TriangleAlert } from "lucide-react";
import { Title } from "@/components/Title";
import { Search } from "@/components/Search";
import { Error } from "@/components/Error";
import { Layout } from "@/components/Layout";

const app = treaty<App>(import.meta.env.VITE_API_URL!);

type SongInfo = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["data"];
type SongInfoError = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["error"];

export async function clientLoader({ params }: Route.LoaderArgs) {
	return { id: params.id };
}

export default function SongInfo({ loaderData }: Route.ComponentProps) {
	const [data, setData] = useState<SongInfo | null>(null);
	const [error, setError] = useState<SongInfoError | null>(null);

	useEffect(() => {
		(async () => {
			const { data, error } = await app.song({ id: loaderData.id }).info.get();
			if (error) {
				console.log(error);
				setError(error);
				return;
			}
			setData(data);
		})();
	}, []);

	if (!data && !error) {
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

	if (error?.status === 404) {
		return (
			<div className="w-screen min-h-screen flex items-center justify-center">
				<Title title="未找到曲目" />
				<div className="max-w-md w-full bg-gray-100 dark:bg-neutral-900 rounded-2xl shadow-lg p-6 flex flex-col gap-4 items-center text-center">
					<div className="w-16 h-16 flex items-center justify-center rounded-full bg-red-500 text-white text-3xl">
						<TriangleAlert size={34} className="-translate-y-0.5" />
					</div>
					<h1 className="text-3xl font-semibold text-neutral-900 dark:text-neutral-100">无法找到曲目</h1>
					<a href={`/song/${loaderData.id}/add`} className="text-secondary-foreground">点此收录</a>
				</div>
			</div>
		);
	}

	if (error) {
		return <Error error={error} />;
	}

	const formatDuration = (duration: number) => {
		const minutes = Math.floor(duration / 60);
		const seconds = duration % 60;
		return `${minutes}:${seconds}`;
	};

	const songNameOnChange = async (e: React.FocusEvent<HTMLHeadingElement, Element>) => {
		const name = e.target.textContent;
		await app.song({ id: loaderData.id }).info.patch({ name: name || undefined });
	};

	return (
		<div className="w-screen min-h-screen relative left-0 top-0 flex justify-center">
			<Title title={data!.name ? data!.name : "未知歌曲名"} />
			<main className="w-full max-sm:mx-6 pt-14 sm:w-xl xl:w-2xl">
				<a href="/">
					<h1 className="text-4xl mb-5">中V档案馆</h1>
				</a>
				<Search />
				{data!.cover && (
					<img
						src={data!.cover}
						referrerPolicy="no-referrer"
						className="w-full aspect-video object-cover rounded-lg mt-6"
					/>
				)}
				<div className="mt-6 flex justify-between">
					<h1 className="text-4xl font-medium" contentEditable={true} onBlur={songNameOnChange}>
						{data!.name ? data!.name : "未知歌曲名"}
					</h1>
					<div className="flex flex-col items-end h-10 whitespace-nowrap">
						<span className="leading-5 text-neutral-800 dark:text-neutral-200">
							{data!.duration ? formatDuration(data!.duration) : "未知时长"}
						</span>
						<span className="text-lg leading-5 text-neutral-800 dark:text-neutral-200 font-bold">
							{data!.producer ? data!.producer : "未知P主"}
						</span>
					</div>
				</div>
			</main>
		</div>
	);
}
