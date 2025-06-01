import useSWRMutation from "swr/mutation";
import { useState } from "react";
import type { CaptchaVerificationRawResponse, CaptchaSessionRawResponse } from "@backend/src/schema";
import { fetcher } from "@/lib/net";
import { computeVdfInWorker } from "@/lib/vdf";

interface UseCaptchaOptions {
	backendURL: string;
	route: string;
}

export function useCaptcha({ backendURL, route }: UseCaptchaOptions) {
	const fullUrl = `${backendURL}/captcha/session`;
	const [isUsed, setIsUsed] = useState(false);

	const { trigger, data, isMutating, error } = useSWRMutation<CaptchaVerificationRawResponse, Error>(
		fullUrl,
		async (url: string) => {
			const sessionRes = await fetcher<CaptchaSessionRawResponse>(url, {
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				data: { route }
			});

			const { g, n, t, id } = sessionRes;
			if (!g || !n || !t || !id) {
				throw new Error("Missing required CAPTCHA parameters");
			}

			const ans = await computeVdfInWorker(BigInt(g), BigInt(n), BigInt(t));

			const resultUrl = new URL(`${backendURL}/captcha/${id}/result`);
			resultUrl.searchParams.set("ans", ans.result.toString());

			const result = await fetcher<CaptchaVerificationRawResponse>(resultUrl.toString());
			setIsUsed(false);
			return result;
		}
	);

	return {
		startCaptcha: trigger,
		captchaResult: data,
		isLoadingCaptcha: isMutating,
		captchaError: error,
		captchaUsed: isUsed,
		setCaptchaUsedState: setIsUsed
	};
}
