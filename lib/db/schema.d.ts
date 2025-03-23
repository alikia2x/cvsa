export interface AllDataType {
	id: number;
	aid: number;
	bvid: string | null;
	description: string | null;
	uid: number | null;
	tags: string | null;
	title: string | null;
	published_at: string | null;
	duration: number;
	created_at: string | null;
}

export interface BiliUserType {
	id: number;
	uid: number;
	username: string;
	desc: string;
	fans: number;
}

export interface VideoSnapshotType {
	id: number;
	created_at: string;
	views: number;
	coins: number;
	likes: number;
	favorites: number;
	shares: number;
	danmakus: number;
	aid: bigint;
	replies: number;
}

export interface LatestSnapshotType {
	aid: number;
	time: number;
	views: number;
	danmakus: number;
	replies: number;
	likes: number;
	coins: number;
	shares: number;
	favorites: number;
}

export interface SnapshotScheduleType {
	id: number;
	aid: number;
	type?: string;
	created_at: string;
	started_at?: string;
	finished_at?: string;
	status: string;
}
