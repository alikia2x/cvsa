"use client";

import { useState } from "react";
import TextField from "@/components/ui/TextField";
import LoadingSpinner from "@/components/icons/LoadingSpinner";
import { computeVdfInWorker } from "@/lib/vdf";
import useSWR from "swr";
import { ApiRequestError } from "@/lib/net";

interface CaptchaSessionResponse {
	g: string;
	n: string;
	t: string;
	id: string;
}

interface CaptchaResultResponse {
	token: string;
}

async function fetcher<JSON = any>(input: RequestInfo, init?: RequestInit): Promise<JSON> {
	const res = await fetch(input, init);
	if (!res.ok) {
		const error = new ApiRequestError("An error occurred while fetching the data.");
		error.response = await res.json();
		error.code = res.status;
		throw error;
	}
	return res.json();
}

interface RegistrationFormProps {
	backendURL: string;
}

const SignUpForm: React.FC<RegistrationFormProps> = ({ backendURL }) => {
	const [username, setUsername] = useState("");
	const [password, setPassword] = useState("");
	const [nickname, setNickname] = useState("");
	const [loading, setLoading] = useState(false);

	const {
		data: captchaSession,
		error: captchaSessionError,
		mutate: createCaptchaSession
	} = useSWR<CaptchaSessionResponse>(
		`${backendURL}/captcha/session`,
		(url) =>
			fetcher(url, {
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify({ route: "POST-/user" })
			}),
		{ revalidateOnFocus: false, revalidateOnReconnect: false }
	);

	const getCaptchaResult = async (id: string, ans: string): Promise<CaptchaResultResponse> => {
		const url = new URL(`${backendURL}/captcha/${id}/result`);
		url.searchParams.set("ans", ans);
		return fetcher<CaptchaResultResponse>(url.toString());
	};

	const register = async () => {
		setLoading(true);
		try {
			if (!captchaSession?.g || !captchaSession?.n || !captchaSession?.t || !captchaSession?.id) {
				console.error("Captcha session data is missing.");
				return;
			}
			const ans = await computeVdfInWorker(
				BigInt(captchaSession.g),
				BigInt(captchaSession.n),
				BigInt(captchaSession.t)
			);
			const captchaResult = await getCaptchaResult(captchaSession.id, ans.result.toString());

			if (!captchaResult.token) {
			}
			// Proceed with user registration using username, password, and nickname
			const registrationUrl = new URL(`${backendURL}/user`);
			const registrationResponse = await fetch(registrationUrl.toString(), {
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify({
					username,
					password,
					nickname,
					// Include the captcha result if needed by the backend
					captchaId: captchaSession.id,
					captchaAnswer: ans.result.toString()
				})
			});

			if (registrationResponse.ok) {
				console.log("Registration successful!");
				// Optionally redirect the user or show a success message
				//router.push("/login"); // Example redirection
			} else {
				console.error("Registration failed:", await registrationResponse.json());
				// Handle registration error
			}
		} catch (error) {
			console.error("Registration process error:", error);
			// Handle general error
		} finally {
			setLoading(false);
		}
	};

	return (
		<form
			className="w-full flex flex-col gap-6"
			onSubmit={async (e) => {
				e.preventDefault();
				await register();
			}}
		>
			<TextField
				labelText="用户名"
				inputText={username}
				onInputTextChange={setUsername}
				maxChar={50}
				supportingText="*必填。用户名是唯一的，不区分大小写。"
			/>
			<TextField
				labelText="密码"
				type="password"
				inputText={password}
				onInputTextChange={setPassword}
				supportingText="*必填。密码至少为 4 个字符。"
				maxChar={120}
			/>
			<TextField
				labelText="昵称"
				inputText={nickname}
				onInputTextChange={setNickname}
				supportingText="昵称可以重复。"
				maxChar={30}
			/>
			<button
				className="bg-primary dark:bg-dark-primary text-on-primary dark:text-dark-on-primary duration-150
        rounded-full hover:bg-on-primary-container hover:dark:bg-dark-on-primary-container mt-2
        flex items-center text-sm leading-5 justify-center h-10 w-full"
				type="submit"
				disabled={loading}
			>
				{!loading ? <span>注册</span> : <LoadingSpinner />}
			</button>
		</form>
	);
};

export default SignUpForm;
