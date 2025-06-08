import { DialogButton, DialogButtonGroup, DialogHeadline, DialogSupportingText } from "@/components/ui/Dialog";
import type { ErrorCode as ResponseErrorCode } from "@cvsa/backend";
import { useTranslations } from "next-intl";

interface ErrorDialogProps {
	children: React.ReactNode;
	closeDialog: () => void;
	errorCode?: ResponseErrorCode;
}

export const ErrorDialog: React.FC<ErrorDialogProps> = ({ children, closeDialog, errorCode }) => {
	const t = useTranslations("backend.error_code");
	return (
		<>
			<DialogHeadline>{errorCode ? t(errorCode) : "错误"}</DialogHeadline>
			<DialogSupportingText>{children}</DialogSupportingText>
			<DialogButtonGroup close={closeDialog}>
				<DialogButton onClick={closeDialog}>关闭</DialogButton>
			</DialogButtonGroup>
		</>
	);
};
