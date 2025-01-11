import axios from "axios";

// 随机选择 User-Agent
function getRandomUserAgent() {
    const userAgents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        // 可以添加更多的 User-Agent 字符串
    ];
    const randomIndex = Math.floor(Math.random() * userAgents.length);
    return userAgents[randomIndex];
}

export function getBiliBiliVideoInfo(bvidORaid?: string | number) {
    const bvid = typeof bvidORaid === "string" ? bvidORaid : undefined;
    const aid = typeof bvidORaid === "number" ? bvidORaid : undefined;
    if (!bvid && !aid) {
        return null;
    }
    const baseURL = "https://api.bilibili.com/x/web-interface/view/detail";
    const headers = {
        'User-Agent': getRandomUserAgent(), // 添加随机 User-Agent
    };

    if (aid) {
        return axios.get(baseURL, {
            params: {
                aid: aid,
            },
            headers: headers, // 将 headers 添加到请求中
        });
    } else {
        return axios.get(baseURL, {
            params: {
                bvid: bvid,
            },
            headers: headers, // 将 headers 添加到请求中
        });
    }
}
