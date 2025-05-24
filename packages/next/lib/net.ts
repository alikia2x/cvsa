export class ApiRequestError extends Error {
	public code: number | undefined;
	public response: unknown | undefined;
	constructor(message: string, res?: unknown, code?: number) {
		super(message);
		this.name = "ApiRequestError";
		this.code = code;
		this.response = res;
	}
}
