"use strict";

export const handler = async (event, context) => {
	const eventObj = JSON.parse(event);
	console.log(`receive event: ${JSON.stringify(eventObj)}`);

	let body = "Missing parameter: URL";
	let statusCode = 400;

	// User-Agent list
	const userAgents = [
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
		"Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
		"Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/89.0",
	];

	// get http request body
	if ("body" in eventObj) {
		body = eventObj.body;
		if (eventObj.isBase64Encoded) {
			body = Buffer.from(body, "base64").toString("utf-8");
		}
	}
	console.log(`receive http body: ${body}`);

	// proxy the URL if it exists in eventObj
	const refererUrl = "https://www.bilibili.com/"; // Replace with your desired referer and origin

	if ("url" in eventObj) {
		try {
			const randomUserAgent = userAgents[Math.floor(Math.random() * userAgents.length)];
			const response = await fetch(eventObj.url, {
				headers: {
					"User-Agent": randomUserAgent,
					"Referer": refererUrl,
				},
			});
			statusCode = response.status;
			body = await response.text();
		} catch (error) {
			statusCode = 500;
			body = `Error fetching URL: ${error.message}`;
		}
	} else if ("urls" in eventObj && Array.isArray(eventObj.urls)) {
		const requests = eventObj.urls.map(async (url) => {
			try {
				const randomUserAgent = userAgents[Math.floor(Math.random() * userAgents.length)];
				const response = await fetch(url, {
					headers: {
						"User-Agent": randomUserAgent,
						"Referer": refererUrl,
					},
				});
				const responseBody = await response.text();
				return {
					statusCode: response.status,
					body: responseBody,
				};
			} catch (error) {
				return {
					statusCode: 500,
					body: `Error fetching URL: ${error.message}`,
				};
			}
		});

		body = await Promise.all(requests);
		statusCode = 200; // Assuming all URLs were processed successfully
	}

	return {
		"statusCode": statusCode,
		"body": JSON.stringify(body),
	};
};
