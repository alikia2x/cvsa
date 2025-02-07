interface BaseResponse<T> {
    code: number;
    message: string;
    ttl: number;
    data: T;
}

export type VideoListResponse = BaseResponse<VideoListData>;
export type VideoTagsResponse = BaseResponse<VideoTagsData>;

type VideoTagsData = VideoTags[];

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
    }
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
    rights: {
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
    },
    owner: {
        mid: number;
        name: string;
        face: string;
    },
    stat: {
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
        dislike: number;
        vt: number;
        vv: number;
    },
    dynamic: string;
    cid: number;
    dimension: {
        width: number;
        height: number;
        rotate: number;
    },
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
