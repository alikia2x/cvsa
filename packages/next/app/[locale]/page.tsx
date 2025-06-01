import { Header } from "@/components/shell/Header";
import { fetcher } from "@/lib/net";
import { UserResponse } from "@backend/src/schema";
import { cookies } from "next/headers";

export default async function Home() {
	const cookieStore = await cookies();
	const sessionID = cookieStore.get("session_id");
	let user: undefined | UserResponse = undefined;
	if (sessionID) {
		try {
			user = await fetcher<UserResponse>(`${process.env.BACKEND_URL}/user/session/${sessionID.value}`);
		} finally {
		}
	}
	return (
		<>
			<Header user={user} />
			<main className="flex flex-col items-center justify-center h-full flex-grow gap-8 px-4">
				<h1 className="text-4xl font-medium text-center">正在施工中……</h1>
				<p>在搜索栏输入BV号或AV号，可以查询目前数据库收集到的信息~</p>
			</main>
		</>
	);
}
