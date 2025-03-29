interface BaseResponse<T> {
	code: number;
	message: string;
	ttl: number;
	data: T;
}

export type VideoListResponse = BaseResponse<VideoListData>;
export type VideoDetailsResponse = BaseResponse<VideoDetailsData>;
export type VideoTagsResponse = BaseResponse<VideoTagsData>;
export type VideoInfoResponse = BaseResponse<VideoInfoData>;
export type MediaListInfoResponse = BaseResponse<MediaListInfoData>;

export type MediaListInfoData = MediaListInfoItem[];


export interface MediaListInfoItem {
	attr: number;
    bvid: string;
	id: number;
	cnt_info: {
		coin: number;
		collect: number;
		danmaku: number;
		play: number;
		reply: number;
		share: number;
		thumb_up: number;
	}
}

interface VideoInfoData {
	bvid: string;
	aid: number;
	copyright: number;
	pic: string;
	title: string;
	pubdate: number;
	ctime: number;
	desc: string;
	desc_v2: string;
	state: number;
	duration: number;
	owner: {
		mid: number;
		name: string;
		face: string;
	};
	stat: VideoStats;
}

interface VideoDetailsData {
	View: {
		bvid: string;
		aid: number;
		videos: number;
		tid: number;
		tid_v2: number;
		tname: string;
		tname_v2: string;
		copyright: number;
		pic: string;
		title: string;
		pubdate: number;
		ctime: number;
		desc: string;
		desc_v2: string;
		state: number;
		duration: number;
		mission_id: number;
		rights: VideoRights;
		owner: {
			mid: number;
			name: string;
			face: string;
		};
		stat: VideoStats;
		argue_info: {
			argue_msg: string;
			argue_type: number;
			argue_link: string;
		};
		dynamic: "";
		cid: number;
		dimension: VideoDimension;
		pages: VideoPage[];
		subtitle: {
			allow_submit: number;
			list: VideoSubTitle[];
		};
		staff: VideoStaff[];
	};
	Card: {
		card: {
			mid: number;
			name: string;
			sex: string;
			face: string;
			fans: number;
			attention: number;
			friend: number;
			sign: string;
			level_info: {
				current_level: number;
			};
		};
		archive_count: number;
		article_count: number;
		follower: number;
		like_num: number;
	};
	Tags: VideoTagsLite[];
}

interface VideoTagsLite {
	tag_id: number;
	tag_name: string;
	music_id: string;
	tag_type: string;
	jump_url: string;
}

type VideoTagsData = VideoTags[];

type VideoStaff = {
	mid: number;
	title: string;
	name: string;
	face: string;
	follower: number;
};

type VideoSubTitle = {
	id: number;
	lan: string;
	lan_doc: string;
	is_lock: number;
	subtitle_url: string;
	type: number;
	id_str: string;
	ai_type: number;
	ai_status: number;
};

type VideoDimension = {
	width: number;
	height: number;
	rotate: number;
};

interface VideoPage {
	cid: number;
	page: number;
	from: string;
	part: string;
	duration: number;
	vid: string;
	weblink: string;
	dimension: VideoDimension;
	first_frame: string;
}

interface VideoTags {
	tag_id: number;
	tag_name: string;
	cover: string;
	head_cover: string;
	content: string;
	short_content: string;
	type: number;
	state: number;
	ctime: number;
	count: {
		view: number;
		use: number;
		atten: number;
	};
	is_atten: number;
	likes: number;
	hates: number;
	attribute: number;
	liked: number;
	hated: number;
	extra_attr: number;
}

interface VideoListData {
	archives: VideoListVideo[];
	page: {
		num: number;
		size: number;
		count: number;
	};
}

type VideoRights = {
	bp: number;
	elec: number;
	download: number;
	movie: number;
	pay: number;
	hd5: number;
	no_reprint: number;
	autoplay: number;
	ugc_pay: number;
	is_cooperation: number;
	ugc_pay_preview: number;
	no_background: number;
	arc_pay: number;
	pay_free_watch: number;
};

type VideoStats = {
	aid: number;
	view: number;
	danmaku: number;
	reply: number;
	favorite: number;
	coin: number;
	share: number;
	now_rank: number;
	his_rank: number;
	like: number;
};

interface VideoListVideo {
	aid: number;
	videos: number;
	tid: number;
	tname: string;
	copyright: number;
	pic: string;
	title: string;
	pubdate: number;
	ctime: number;
	desc: string;
	state: number;
	duration: number;
	mission_id?: number;
	rights: VideoRights;
	owner: {
		mid: number;
		name: string;
		face: string;
	};
	stat: VideoStats;
	dynamic: string;
	cid: number;
	dimension: VideoDimension;
	season_id?: number;
	short_link_v2: string;
	first_frame: string;
	pub_location: string;
	cover43: string;
	tidv2: number;
	tname_v2: string;
	bvid: string;
	season_type: number;
	is_ogv: number;
	ovg_info: string | null;
	rcmd_season: string;
	enable_vt: number;
	ai_rcmd: null | string;
}
