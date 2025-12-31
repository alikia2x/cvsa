export interface Root {
	version: number;
	ranknum: number;
	url: string;
	coverurl: string;
	pubdate: string;
	generate_time: string;
	generate_timestamp: number;
	collect_start_time: string;
	collect_end_time: string;
	collect_start_time_timestamp: number;
	collect_end_time_timestamp: number;
	main_rank: MainRank[];
	second_rank: SecondRank[];
	super_hit: SuperHit[];
	pick_up: PickUp[];
	oth_pickup: any[];
	Vocaloid_pick_up: VocaloidPickUp[];
	"history-1-year": History1Year[];
	"history-10-year": History10Year[];
	ed: Ed[];
	op: Op[];
	statistic: Statistic;
	thanks_list: any[];
}

export interface MainRank {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: any;
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
	referSource: ReferSource;
	rank: number;
	ext_rank: ExtRank;
}

export interface ReferSource {
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
}

export interface ExtRank {
	vocaloid?: number;
}

export interface SecondRank {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
	referSource: ReferSource2;
	rank: number;
	ext_rank: ExtRank2;
}

export interface ReferSource2 {
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
}

export interface ExtRank2 {
	vocaloid?: number;
}

export interface SuperHit {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
	referSource: ReferSource3;
	superHit_times: number;
	rank: string;
}

export interface ReferSource3 {
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
}

export interface PickUp {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
	referSource: ReferSource4;
	rank: number;
	ext_rank: ExtRank4;
}

export interface ReferSource4 {
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
}

export interface ExtRank4 {
	vocaloid?: number;
}

export interface VocaloidPickUp {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
	referSource: ReferSource5;
	rank: number;
	ext_rank: ExtRank5;
}

export interface ReferSource5 {
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
}

export interface ExtRank5 {
	vocaloid: number;
}

export interface History1Year {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
	referSource: ReferSource6;
	rank: number;
	ext_rank: ExtRank6;
}

export interface ReferSource6 {
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
}

export interface ExtRank6 {
	vocaloid?: number;
}

export interface History10Year {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
	referSource: ReferSource7;
	rank: number;
}

export interface ReferSource7 {
	point: number;
	play: number;
	coin: number;
	comment: number;
	danmaku: number;
	favorite: number;
	like: number;
	share: number;
}

export interface Ed {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
}

export interface Op {
	url: string;
	avid: string;
	coverurl: string;
	title: string;
	pubdate: string;
}

export interface Statistic {
	diff: Diff;
	total_collect_count: number;
	new_video_count: number;
	new_in_rank_count: number;
	new_in_mainrank_count: number;
	pick_up_count: number;
	oth_pick_up_count: number;
	new_vc_in_rank_count: number;
	new_vc_in_mainrank_count: number;
	vc_in_rank_count: number;
	vc_in_mainrank_count: number;
	new_sv_in_rank_count: number;
	new_sv_in_mainrank_count: number;
	sv_in_rank_count: number;
	sv_in_mainrank_count: number;
	new_ace_in_rank_count: number;
	new_ace_in_mainrank_count: number;
	ace_in_rank_count: number;
	ace_in_mainrank_count: number;
}

export interface Diff {
	total_play: number;
	new_video_count: number;
	new_in_rank_count: number;
	new_in_mainrank_count: number;
	new_vc_in_rank_count: number;
	new_vc_in_mainrank_count: number;
	vc_in_rank_count: number;
	vc_in_mainrank_count: number;
	new_sv_in_rank_count: number;
	new_sv_in_mainrank_count: number;
	sv_in_rank_count: number;
	sv_in_mainrank_count: number;
	new_ace_in_rank_count: number;
	new_ace_in_mainrank_count: number;
	ace_in_rank_count: number;
	ace_in_mainrank_count: number;
}

const aids = new Set<string>();
const f = Bun.file("evocalrank.json");

for (let i = 699; i >= 520; i--) {
	const url = `https://www.evocalrank.com/data/rank_data/${i}.json`;
	const response = await fetch(url);
	const data = await response.json() as Partial<Root>;
	if (data.main_rank) {
        for (const item of data.main_rank) {
            if (item.avid) {
                aids.add(item.avid);
            }
        }
	}
    if (data.second_rank) {
        for (const item of data.second_rank) {
            if (item.avid) {
                aids.add(item.avid);
            }
        }
    }
    if (data.pick_up) {
        for (const item of data.pick_up) {
            if (item.avid) {
                aids.add(item.avid);
            }
        }
    }
    if (data.super_hit) {
        for (const item of data.super_hit) {
            if (item.avid) {
                aids.add(item.avid);
            }
        }
    }
    if (data.Vocaloid_pick_up) {
        for (const item of data.Vocaloid_pick_up) {
            if (item.avid) {
                aids.add(item.avid);
            }
        }
    }
    if (data.ed) {
        for (const item of data.ed) {
            if (item.avid) {
                aids.add(item.avid);
            }
        }
    }
    if (data.op) {
        for (const item of data.op) {
            if (item.avid) {
                aids.add(item.avid);
            }
        }
    }

    const serialized = JSON.stringify([...aids], null, 4);
    await f.write(serialized);
    console.log(`${i} ${aids.size}`);
}
