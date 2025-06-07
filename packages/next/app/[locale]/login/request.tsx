import { Dispatch, JSX, SetStateAction } from "react";
import { ApiRequestError, fetcher } from "@/lib/net";
import type { CaptchaVerificationRawResponse, ErrorResponse, SignUpResponse } from "@cvsa/backend";
import { Link } from "@/i18n/navigation";
import { LocalizedMessage } from "./LoginForm";
import { ErrorDialog } from "@/components/utils/ErrorDialog";
import { string, object, ValidationError, setLocale } from "yup";

setLocale({
	mixed: {
		default: "yup_errors.field_invalid",
		required: () => ({ key: "yup_errors.field_required" })
	},
	string: {
		min: ({ min }) => ({ key: "yup_errors.field_too_short", values: { min } }),
		max: ({ max }) => ({ key: "yup_errors.field_too_big", values: { max } })
	}
});

interface LoginFormData {
	username: string;
	password: string;
}

const FormSchema = object().shape({
	username: string().required().max(50),
	password: string().required().min(4).max(120)
});

const validateForm = async (
	data: LoginFormData,
	setShowDialog: Dispatch<SetStateAction<boolean>>,
	setDialogContent: Dispatch<SetStateAction<JSX.Element>>,
	translateErrorMessage: (item: LocalizedMessage | string, path?: string) => string
): Promise<LoginFormData | null> => {
	const { username: usernameInput, password: passwordInput } = data;
	try {
		const formData = await FormSchema.validate({
			username: usernameInput,
			password: passwordInput
		});
		return {
			username: formData.username,
			password: formData.password
		};
	} catch (e) {
		if (!(e instanceof ValidationError)) {
			return null;
		}
		setShowDialog(true);
		setDialogContent(
			<ErrorDialog closeDialog={() => setShowDialog(false)}>
				<p>{translateErrorMessage(e.errors[0], e.path)}</p>
			</ErrorDialog>
		);
		return null;
	}
};

interface RequestSignUpArgs {
	data: LoginFormData;
	setShowDialog: Dispatch<SetStateAction<boolean>>;
	setDialogContent: Dispatch<SetStateAction<JSX.Element>>;
	translateErrorMessage: (item: LocalizedMessage | string, path?: string) => string;
	setCaptchaUsedState: Dispatch<SetStateAction<boolean>>;
	captchaResult: CaptchaVerificationRawResponse | undefined;
	t: any;
}

export const requestLogin = async (url: string, { arg }: { arg: RequestSignUpArgs }) => {
	const { data, setShowDialog, setDialogContent, translateErrorMessage, setCaptchaUsedState, captchaResult, t } = arg;
	const res = await validateForm(data, setShowDialog, setDialogContent, translateErrorMessage);
	if (!res) {
		return;
	}
	const { username, password } = res;

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
				password: password
			}
		});
		return registrationResponse;
	} catch (error) {
		if (error instanceof ApiRequestError && error.response) {
			const res = error.response as ErrorResponse;
			setShowDialog(true);
			setDialogContent(
				<ErrorDialog closeDialog={() => setShowDialog(false)} errorCode={res.code}>
					<p>
						无法登录：
						<span>
							{res.i18n
								? t.rich(res.i18n.key, {
										...res.i18n.values,
										support: (chunks: string) => <Link href="/support">{chunks}</Link>
									})
								: res.message}
						</span>
					</p>
				</ErrorDialog>
			);
		} else if (error instanceof Error) {
			setShowDialog(true);
			setDialogContent(
				<ErrorDialog closeDialog={() => setShowDialog(false)}>
					<p>无法登录。</p>
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
					<p>无法登录。</p>
					<p>
						错误信息： <br />
						<pre className="break-all">{JSON.stringify(error)}</pre>
					</p>
				</ErrorDialog>
			);
		}
	}
};
