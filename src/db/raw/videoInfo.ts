export async function getBiliBiliVideoInfo(bvidORaid?: string | number, region: string = "hangzhou") {
    const bvid = typeof bvidORaid === "string" ? bvidORaid : undefined;
    const aid = typeof bvidORaid === "number" ? bvidORaid : undefined;

    const baseURL = "https://api.bilibili.com/x/web-interface/view/detail";
    const urlObject = new URL(baseURL);

    if (aid) {
        urlObject.searchParams.append("aid", aid.toString());
        const finalURL = urlObject.toString();
        return await proxyRequestWithRegion(finalURL, region);
    } else if (bvid) {
        urlObject.searchParams.append("bvid", bvid);
        const finalURL = urlObject.toString();
        return await proxyRequestWithRegion(finalURL, region);
    } else {
        return null;
    }
}

async function proxyRequestWithRegion(url: string, region: string): Promise<any | null> {
    const td = new TextDecoder();
    const p = await new Deno.Command("aliyun", {
        args: [
            "fc",
            "POST",
            `/2023-03-30/functions/proxy-${region}/invocations`,
            "--qualifier",
            "LATEST",
            "--header",
            "Content-Type=application/json;x-fc-invocation-type=Sync;x-fc-log-type=None;",
            "--body",
            JSON.stringify({url: url}),
            "--profile",
            `CVSA-${region}`,
        ],
    }).output();
    try {
        const out = td.decode(p.stdout);
        const rawData = JSON.parse(out);
        if (rawData.statusCode !== 200) {
            console.error(`Error proxying request ${url} to ${region} , statusCode: ${rawData.statusCode}`);
            return null;
        }
        else {
            return JSON.parse(rawData.body);
        }
    }
    catch (e){
        console.error(`Error proxying requestt ${url} to ${region}: ${e}`);
        return null;
    }
}