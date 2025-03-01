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
