type ErrorCode = "INVALID_QUERY_PARAMS" | "UNKNOWN_ERR" | "INVALID_PAYLOAD" | "INVALID_FORMAT" | "BODY_TOO_LARGE";

export interface ErrorResponse<E> {
	code: ErrorCode;
	message: string;
	errors: E[];
}

export interface StatusResponse {
	message: string;
}
