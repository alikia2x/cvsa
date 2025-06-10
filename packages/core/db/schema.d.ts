export interface BiliUserType {
	id: number;
	uid: number;
	username: string;
	desc: string;
	fans: number;
}

export interface VideoSnapshotType {
	id: number;
	created_at: Date;
	views: number;
	coins: number;
	likes: number;
	favorites: number;
	shares: number;
	danmakus: number;
	aid: number;
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
	created_at: Date;
	started_at?: Date;
	finished_at?: Date;
	status: string;
}

export interface UserType {
	id: number;
	username: string;
	nickname: string | null;
	password: string;
	unq_id: string;
	role: string;
}

export interface BiliVideoMetadataType {
	id: number;
	aid: number;
	bvid: string | null;
	description: string | null;
	uid: number | null;
	tags: string | null;
	title: string | null;
	published_at: Date | null;
	duration: number | null;
	created_at: Date;
	status: number;
	cover_url: string | null;
}
