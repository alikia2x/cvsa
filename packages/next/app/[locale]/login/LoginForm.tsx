"use client";

import { useEffect, useState } from "react";
import TextField from "@/components/ui/TextField";
import LoadingSpinner from "@/components/icons/LoadingSpinner";
import { Portal } from "@/components/utils/Portal";
import { Dialog } from "@/components/ui/Dialog";
import { setLocale } from "yup";
import { useTranslations } from "next-intl";
import { useCaptcha } from "@/components/hooks/useCaptcha";
import useSWRMutation from "swr/mutation";
import { FilledButton } from "@/components/ui/Buttons/FilledButton";
import { ApiRequestError } from "@/lib/net";
import { useRouter } from "next/navigation";
import { requestLogin } from "./request";
import { ErrorDialog } from "@/components/utils/ErrorDialog";

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

export interface LocalizedMessage {
	key: string;
	values: {
		[key: string]: number | string;
	};
}

interface RegistrationFormProps {
	backendURL: string;
}

const SignUpForm: React.FC<RegistrationFormProps> = ({ backendURL }) => {
	const [usernameInput, setUsername] = useState("");
	const [passwordInput, setPassword] = useState("");
	const [showDialog, setShowDialog] = useState(false);
	const [dialogContent, setDialogContent] = useState(<></>);
	const [isLoading, setLoading] = useState(false);
	const t = useTranslations("");
	const { startCaptcha, captchaResult, captchaUsed, setCaptchaUsedState, captchaError } = useCaptcha({
		backendURL,
		route: "POST-/user"
	});
	const { trigger } = useSWRMutation(`${backendURL}/login/session`, requestLogin);
	const router = useRouter();

	const translateErrorMessage = (item: LocalizedMessage | string, path?: string) => {
		if (typeof item === "string") {
			return item;
		}
		return t(`${item.key}`, { ...item.values, field: path ? t(path) : "" });
	};

	const register = async () => {
		try {
			if (captchaUsed || !captchaResult) {
				await startCaptcha();
			}

			const result = await trigger({
				data: {
					username: usernameInput,
					password: passwordInput
				},
				setShowDialog,
				captchaResult,
				setCaptchaUsedState,
				translateErrorMessage,
				setDialogContent,
				t
			});
			if (result) {
				router.push("/");
			}
		} finally {
			setLoading(false);
		}
	};

	useEffect(() => {
		if (!captchaError || captchaError === undefined) return;
		const err = captchaError as ApiRequestError;
		setShowDialog(true);
		if (err.code && err.code == -1) {
			setDialogContent(
				<ErrorDialog closeDialog={() => setShowDialog(false)}>
					<p>无法连接到服务器，请检查你的网络连接后重试。</p>
				</ErrorDialog>
			);
		}
	}, [captchaError]);

	useEffect(() => {
		startCaptcha();
	}, []);

	return (
		<form
			className="w-full flex flex-col gap-6"
			onSubmit={async (e) => {
				setLoading(true);
				e.preventDefault();
				await register();
			}}
		>
			<TextField labelText="用户名" inputText={usernameInput} onInputTextChange={setUsername} />
			<TextField labelText="密码" type="password" inputText={passwordInput} onInputTextChange={setPassword} />
			<FilledButton type="submit" disabled={isLoading}>
				{isLoading ? <LoadingSpinner /> : <span>登录</span>}
			</FilledButton>
			<Portal>
				<Dialog show={showDialog}>{dialogContent}</Dialog>
			</Portal>
		</form>
	);
};

export default SignUpForm;
