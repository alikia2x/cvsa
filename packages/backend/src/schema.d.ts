type ErrorCode =
	| "INVALID_QUERY_PARAMS"
	| "UNKNOWN_ERROR"
	| "INVALID_PAYLOAD"
	| "INVALID_FORMAT"
	| "INVALID_HEADER"
	| "BODY_TOO_LARGE"
	| "UNAUTHORIZED"
	| "INVALID_CREDENTIALS"
	| "ENTITY_NOT_FOUND"
	| "SERVER_ERROR";

export interface ErrorResponse<E=string> {
	code: ErrorCode;
	message: string;
	errors?: E[];
}

export interface StatusResponse {
	message: string;
}
