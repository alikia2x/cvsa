"use client";

import { useEffect, useState } from "react";
import TextField from "@/components/ui/TextField";
import LoadingSpinner from "@/components/icons/LoadingSpinner";
import { ApiRequestError } from "@/lib/net";
import { Portal } from "@/components/utils/Portal";
import { Dialog, DialogButton, DialogButtonGroup, DialogHeadline, DialogSupportingText } from "@/components/ui/Dialog";
import { FilledButton } from "@/components/ui/Buttons/FilledButton";
import { string, object, ValidationError } from "yup";
import { setLocale } from "yup";
import { useTranslations } from "next-intl";
import type { ErrorResponse } from "@backend/src/schema";
import Link from "next/link";
import { useCaptcha } from "@/components/hooks/useCaptcha";

setLocale({
	mixed: {
		default: "field_invalid",
		required: () => ({ key: "field_required" })
	},
	string: {
		min: ({ min }) => ({ key: "field_too_short", values: { min } }),
		max: ({ max }) => ({ key: "field_too_big", values: { max } })
	}
});

interface LocalizedMessage {
	key: string;
	values: {
		[key: string]: number | string;
	};
}

const FormSchema = object().shape({
	username: string().required().max(50),
	password: string().required().min(4).max(120),
	nickname: string().optional().max(30)
});

interface RegistrationFormProps {
	backendURL: string;
}

interface ErrorDialogProps {
	children: React.ReactNode;
	closeDialog: () => void;
}

const ErrorDialog: React.FC<ErrorDialogProps> = ({ children, closeDialog }) => {
	return (
		<>
			<DialogHeadline>错误</DialogHeadline>
			<DialogSupportingText>{children}</DialogSupportingText>
			<DialogButtonGroup>
				<DialogButton onClick={closeDialog}>关闭</DialogButton>
			</DialogButtonGroup>
		</>
	);
};

const SignUpForm: React.FC<RegistrationFormProps> = ({ backendURL }) => {
	const [usernameInput, setUsername] = useState("");
	const [passwordInput, setPassword] = useState("");
	const [nicknameInput, setNickname] = useState("");
	const [loading, setLoading] = useState(false);
	const [showDialog, setShowDialog] = useState(false);
	const [dialogContent, setDialogContent] = useState(<></>);
	const t = useTranslations("");
	const { startCaptcha, captchaResult } = useCaptcha({
		backendURL,
		route: "POST-/user"
	});

	const translateErrorMessage = (item: LocalizedMessage | string, path?: string) => {
		if (typeof item === "string") {
			return item;
		}
		return t(`yup_errors.${item.key}`, { ...item.values, field: path ? t(path) : "" });
	};

	const register = async () => {
		let username: string | undefined;
		let password: string | undefined;
		let nickname: string | undefined;
		try {
			const formData = await FormSchema.validate(
				{
					username: usernameInput,
					password: passwordInput,
					nickname: nicknameInput
				},
				{ abortEarly: false }
			);
			username = formData.username;
			password = formData.password;
			nickname = formData.nickname;
		} catch (e) {
			if (!(e instanceof ValidationError)) {
				return;
			}
			console.log(JSON.parse(JSON.stringify(e)));
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
			return;
		}

		setLoading(true);
		try {
			if (!captchaResult || !captchaResult.token) {
				await startCaptcha();
			}
			// Proceed with user registration using username, password, and nickname
			const registrationUrl = new URL(`${backendURL}/user`);
			const registrationResponse = await fetch(registrationUrl.toString(), {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${captchaResult!.token}`
				},
				body: JSON.stringify({
					username: username,
					password: password,
					nickname: nickname
				})
			});

			if (registrationResponse.ok) {
				console.log("Registration successful!");
				// Optionally redirect the user or show a success message
				//router.push("/login"); // Example redirection
			} else {
				const res: ErrorResponse = await registrationResponse.json();
				setShowDialog(true);
				setDialogContent(
					<ErrorDialog closeDialog={() => setShowDialog(false)}>
						<p>无法为你注册账户。</p>
						<p>错误码: {res.code}</p>
						<p>
							错误信息: <br />
							{res.i18n ? t(res.i18n.key, { ...res.i18n.values }) : res.message}
						</p>
					</ErrorDialog>
				);
			}
		} catch (error) {
			if (error instanceof ApiRequestError) {
				const res = error.response as ErrorResponse;
				setShowDialog(true);
				setDialogContent(
					<ErrorDialog closeDialog={() => setShowDialog(false)}>
						<p>无法为你注册账户。</p>
						<p>错误码: {res.code}</p>
						<p>
							错误信息: <br />
							{res.i18n
								? t.rich(res.i18n.key, {
										...res.i18n.values,
										support: (chunks) => <Link href="/support">{chunks}</Link>
									})
								: res.message}
						</p>
					</ErrorDialog>
				);
			}
		} finally {
			setLoading(false);
		}
	};

	useEffect(() => {
		if (startCaptcha) {
			startCaptcha();
		}
	}, [startCaptcha]);

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
				inputText={usernameInput}
				onInputTextChange={setUsername}
				maxChar={50}
				supportingText="*必填。用户名是唯一的，不区分大小写。"
			/>
			<TextField
				labelText="密码"
				type="password"
				inputText={passwordInput}
				onInputTextChange={setPassword}
				supportingText="*必填。密码至少为 4 个字符。"
				maxChar={120}
			/>
			<TextField
				labelText="昵称"
				inputText={nicknameInput}
				onInputTextChange={setNickname}
				supportingText="昵称可以重复。"
				maxChar={30}
			/>
			<FilledButton type="submit" disabled={loading}>
				{!loading ? <span>注册</span> : <LoadingSpinner />}
			</FilledButton>
			<Portal>
				<Dialog show={showDialog}>{dialogContent}</Dialog>
			</Portal>
		</form>
	);
};

export default SignUpForm;
