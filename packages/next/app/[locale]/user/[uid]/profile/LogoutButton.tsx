"use client";

import { TextButton } from "@/components/ui/Buttons/TextButton";
import { Dialog, DialogButton, DialogButtonGroup, DialogHeadline, DialogSupportingText } from "@/components/ui/Dialog";
import { Portal } from "@/components/utils/Portal";
import { useRouter } from "@/i18n/navigation";
import { useState } from "react";

export const LogoutButton: React.FC = () => {
	const [showDialog, setShowDialog] = useState(false);
	const router = useRouter();
	return (
		<>
			<TextButton className="font-medium" onClick={() => setShowDialog(true)}>
				登出
			</TextButton>
			<Portal>
				<Dialog show={showDialog}>
					<DialogHeadline>确认登出</DialogHeadline>
					<DialogSupportingText>确认要退出登录吗？</DialogSupportingText>
					<DialogButtonGroup close={() => setShowDialog(false)}>
						<DialogButton onClick={() => setShowDialog(false)}>取消</DialogButton>
						<DialogButton
							onClick={async () => {
								try {
									await fetch("/logout", {
										method: "POST"
									});
									router.push("/");
								} finally {
									setShowDialog(false);
								}
							}}
						>
							确认
						</DialogButton>
					</DialogButtonGroup>
				</Dialog>
			</Portal>
		</>
	);
};
