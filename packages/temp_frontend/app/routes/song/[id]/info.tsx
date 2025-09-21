import useSWR from "swr";
import type { Route } from "./+types/info";

const API_URL = "https://api.projectcvsa.com";

export async function clientLoader({ params }: Route.LoaderArgs) {
	return { id: params.id };
}

export default function SongInfo({ loaderData }: Route.ComponentProps) {
	const { data, error, isLoading } = useSWR(`${API_URL}/video/${loaderData.id}/info`, async (url) => {
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error("Failed to fetch song info");
		}
		return response.json();
	});

	if (isLoading) return <div>加载中...</div>;
	if (error) return <div>错误: {error.message}</div>;
	if (!data) return <div>暂无数据</div>;

	return (
		<div>
			<h1>歌曲信息</h1>
			<pre>{JSON.stringify(data, null, 2)}</pre>
		</div>
	);
}
