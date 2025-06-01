import { Dispatch, JSX, SetStateAction } from "react";
import { ApiRequestError, fetcher } from "@/lib/net";
import type { CaptchaVerificationRawResponse, ErrorResponse, SignUpResponse } from "@backend/src/schema";
import { Link } from "@/i18n/navigation";
import { LocalizedMessage } from "./SignUpForm";
import { ErrorDialog } from "./ErrorDialog";
import { string, object, ValidationError } from "yup";

interface SignUpFormData {
	username: string;
	password: string;
	nickname?: string;
}

const FormSchema = object().shape({
	username: string().required().max(50),
	password: string().required().min(4).max(120),
	nickname: string().optional().max(30)
});

const validateForm = async (
	data: SignUpFormData,
	setShowDialog: Dispatch<SetStateAction<boolean>>,
	setDialogContent: Dispatch<SetStateAction<JSX.Element>>,
	translateErrorMessage: (item: LocalizedMessage | string, path?: string) => string
): Promise<SignUpFormData | null> => {
	const { username: usernameInput, password: passwordInput, nickname: nicknameInput } = data;
	try {
		const formData = await FormSchema.validate(
			{
				username: usernameInput,
				password: passwordInput,
				nickname: nicknameInput
			},
			{ abortEarly: false }
		);
		return {
			username: formData.username,
			password: formData.password,
			nickname: formData.nickname
		};
	} catch (e) {
		if (!(e instanceof ValidationError)) {
			return null;
		}
		setShowDialog(true);
		setDialogContent(
			<ErrorDialog closeDialog={() => setShowDialog(false)}>
				<p>注册信息填写有误，请检查后重新提交。</p>
				<span>错误信息: </span>
				<br />
				<ol className="list-decimal list-inside">
					{e.errors.map((item, i) => {
						return <li key={i}>{translateErrorMessage(item, e.inner[i].path)}</li>;
					})}
				</ol>
			</ErrorDialog>
		);
		return null;
	}
};

interface RequestSignUpArgs {
	data: SignUpFormData;
	setShowDialog: Dispatch<SetStateAction<boolean>>;
	setDialogContent: Dispatch<SetStateAction<JSX.Element>>;
	translateErrorMessage: (item: LocalizedMessage | string, path?: string) => string;
	setCaptchaUsedState: Dispatch<SetStateAction<boolean>>;
	captchaResult: CaptchaVerificationRawResponse | undefined;
	t: any;
}

export const requestSignUp = async (url: string, { arg }: { arg: RequestSignUpArgs }) => {
	const { data, setShowDialog, setDialogContent, translateErrorMessage, setCaptchaUsedState, captchaResult, t } = arg;
	const res = await validateForm(data, setShowDialog, setDialogContent, translateErrorMessage);
	if (!res) {
		return;
	}
	const { username, nickname, password } = res;

	try {
		if (!captchaResult) {
			const err = new ApiRequestError("Cannot get captcha result");
			err.response = {
				code: "UNKNOWN_ERROR",
				message: "Cannot get captch verifiction result",
				i18n: {
					key: "captcha_failed_to_get"
				}
			} as ErrorResponse;
			throw err;
		}
		setCaptchaUsedState(true);
		const registrationResponse = await fetcher<SignUpResponse>(url, {
			method: "POST",
			withCredentials: true,
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${captchaResult!.token}`
			},
			data: {
				username: username,
				password: password,
				nickname: nickname
			}
		});
		return registrationResponse;
	} catch (error) {
		if (error instanceof ApiRequestError && error.response) {
			const res = error.response as ErrorResponse;
			setShowDialog(true);
			setDialogContent(
				<ErrorDialog closeDialog={() => setShowDialog(false)} errorCode={res.code}>
					<p>无法为你注册账户。</p>
					<p>
						错误信息: <br />
						{res.i18n
							? t.rich(res.i18n.key, {
									...res.i18n.values,
									support: (chunks: string) => <Link href="/support">{chunks}</Link>
								})
							: res.message}
					</p>
				</ErrorDialog>
			);
		} else if (error instanceof Error) {
			setShowDialog(true);
			setDialogContent(
				<ErrorDialog closeDialog={() => setShowDialog(false)}>
					<p>无法为你注册账户。</p>
					<p>
						错误信息：
						<br />
						{error.message}
					</p>
				</ErrorDialog>
			);
		} else {
			setShowDialog(true);
			setDialogContent(
				<ErrorDialog closeDialog={() => setShowDialog(false)} errorCode="UNKNOWN_ERROR">
					<p>无法为你注册账户。</p>
					<p>
						错误信息： <br />
						<pre className="break-all">{JSON.stringify(error)}</pre>
					</p>
				</ErrorDialog>
			);
		}
	}
};
