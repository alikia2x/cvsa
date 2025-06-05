export type ErrorCode =
	| "INVALID_QUERY_PARAMS"
	| "UNKNOWN_ERROR"
	| "INVALID_PAYLOAD"
	| "INVALID_FORMAT"
	| "INVALID_HEADER"
	| "BODY_TOO_LARGE"
	| "UNAUTHORIZED"
	| "INVALID_CREDENTIALS"
	| "ENTITY_NOT_FOUND"
	| "SERVER_ERROR"
	| "RATE_LIMIT_EXCEEDED"
	| "ENTITY_EXISTS";

export interface ErrorResponse<E = string> {
	code: ErrorCode;
	message: string;
	errors: E[] = [];
	i18n?: {
		key: string;
		values?: {
			[key: string]: string | number | Date;
		};
	};
}

export interface StatusResponse {
	message: string;
}

export type CaptchaSessionResponse = ErrorResponse | CaptchaSessionRawResponse;

interface CaptchaSessionRawResponse {
	success: boolean;
	id: string;
	g: string;
	n: string;
	t: number;
}

export interface SignUpResponse {
    username: string;
    token: string;
}

export interface UserResponse {
    username: string;
    nickname: string | null;
    role: string;
}

export type CaptchaVerificationRawResponse = {
	token: string;
}

export type CaptchaVerificationResponse =
	| ErrorResponse
	| CaptchaVerificationRawResponse;
