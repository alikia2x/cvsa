import useSWRMutation from "swr/mutation";
import type { ErrorResponse, CaptchaSessionResponse, CaptchaVerificationRawResponse } from "@backend/src/schema";
import { fetcher } from "@/lib/net";
import { computeVdfInWorker } from "@/lib/vdf";

interface UseCaptchaOptions {
	backendURL: string;
	route: string;
}

function isErrResponse(res: ErrorResponse | object): res is ErrorResponse {
	return (res as ErrorResponse).errors !== undefined;
}

export function useCaptcha({ backendURL, route }: UseCaptchaOptions) {
	const fullUrl = `${backendURL}/captcha/session`;

	const { trigger, data, isMutating, error } = useSWRMutation<CaptchaVerificationRawResponse, Error>(
		fullUrl,
		async (url: string) => {
			const sessionRes = await fetcher<CaptchaSessionResponse>(url, {
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				data: { route }
			});

			if (isErrResponse(sessionRes)) {
				throw new Error(sessionRes.message || "Failed to get captcha session");
			}

			const { g, n, t, id } = sessionRes;
			if (!g || !n || !t || !id) {
				throw new Error("Missing required CAPTCHA parameters");
			}

			const ans = await computeVdfInWorker(BigInt(g), BigInt(n), BigInt(t));

			const resultUrl = new URL(`${backendURL}/captcha/${id}/result`);
			resultUrl.searchParams.set("ans", ans.result.toString());

			const result = await fetcher<CaptchaVerificationRawResponse>(resultUrl.toString());
			return result;
		}
	);

	return {
		startCaptcha: trigger,
		captchaResult: data,
		isLoadingCaptcha: isMutating,
		captchaError: error
	};
}
