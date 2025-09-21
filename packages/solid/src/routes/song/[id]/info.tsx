import { Layout } from "~/components/layout";
import { LeftSideBar } from "~/components/song/LeftSideBar";
import { RightSideBar } from "~/components/song/RightSideBar";
import { Content } from "~/components/song/Content";
import { createAsync, query, RouteDefinition, useParams } from "@solidjs/router";
import { dbMain } from "~/drizzle";
import { bilibiliMetadata, songs } from "~db/main/schema";
import { eq } from "drizzle-orm";

const getVideoAID = async (id: string) => {
	"use server";
	if (id.startsWith("av")) {
		return parseInt(id.slice(2));
	} else if (id.startsWith("BV")) {
		const data = await dbMain
			.select()
			.from(bilibiliMetadata)
			.where(eq(bilibiliMetadata.bvid, id));
		return data[0].aid;
	}
	else {
		return null;
	}
};

const findSongIDFromAID = async (aid: number) => {
	"use server";
	const data = await dbMain.select({
		id: songs.id,
	}).from(songs).where(eq(songs.aid, aid)).limit(1);
	return data[0].id;
}

const getSongInfo = query(async (songID: number) => {
	"use server";
	const data = await dbMain.select().from(songs).where(eq(songs.id, songID));
	return data[0] || null;
}, "songs");

const getSongInfoFromID = query(async (id: string) => {
	"use server";
	const aid = await getVideoAID(id);
	if (!aid && parseInt(id)) {
		return getSongInfo(parseInt(id));
	}
	else if (!aid) {
		return null;
	}
	const songID = await findSongIDFromAID(aid);
	return getSongInfo(songID);
}, "songsRaw")

export const route = {
	preload: ({ params }) => getSongInfoFromID(params.id)
} satisfies RouteDefinition;


export default function Info() {
	const params = useParams();
	const info = createAsync(() => getSongInfoFromID(params.id));
	return (
		<Layout>
			<title>尘海绘仙缘 - 歌曲信息 - 中 V 档案馆</title>
			<div
				class="pt-8 w-full sm:w-120 sm:mx-auto lg:w-full 2xl:w-360 lg:grid lg:grid-cols-[1fr_560px_1fr]
					xl:grid-cols-[1fr_648px_1fr]"
			>
				<nav class="top-32 hidden lg:block pb-12 px-6 self-start sticky">
					<LeftSideBar />
				</nav>
				<main class="mb-24">
					<Content data={info() || null}/>
				</main>
				<div class="top-32 hidden lg:flex self-start sticky flex-col pb-12 px-6">
					<RightSideBar />
				</div>
			</div>
		</Layout>
	);
}
