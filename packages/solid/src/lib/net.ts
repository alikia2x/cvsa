import axios, { AxiosError, AxiosRequestConfig, AxiosResponse, Method } from "axios";

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

type HttpMethod = Extract<Method, "GET" | "POST" | "PUT" | "DELETE" | "PATCH">;

const httpMethods = {
	get: axios.get,
	post: axios.post,
	put: axios.put,
	delete: axios.delete,
	patch: axios.patch
};

export function fetcher(url: string): Promise<unknown>;
export function fetcher<JSON = unknown>(
	url: string,
	init?: Omit<AxiosRequestConfig, "method"> & { method?: Exclude<HttpMethod, "DELETE"> }
): Promise<JSON>;
export function fetcher(
	url: string,
	init: Omit<AxiosRequestConfig, "method"> & { method: "DELETE" }
): Promise<AxiosResponse>;

export async function fetcher<JSON = unknown>(
	url: string,
	init?: Omit<AxiosRequestConfig, "method"> & { method?: HttpMethod }
): Promise<JSON | AxiosResponse<any, any>> {
	const { method = "get", data, ...config } = init || {};

	const fullConfig: AxiosRequestConfig = {
		method,
		...config,
		timeout: 10000
	};

	try {
		const m = method.toLowerCase() as keyof typeof httpMethods;
		if (["post", "patch", "put"].includes(m)) {
			const response = await httpMethods[m](url, data, fullConfig);
			return response.data;
		} else if (m === "delete") {
			return await axios.delete(url, fullConfig);
		} else {
			const response = await httpMethods[m](url, fullConfig);
			return response.data;
		}
	} catch (error) {
		const axiosError = error as AxiosError;

		if (axiosError.response) {
			const { status, data } = axiosError.response;
			throw new ApiRequestError(`HTTP error! status: ${status}`, data, status);
		} else if (axiosError.request) {
			throw new ApiRequestError("No response received", undefined, -1);
		} else {
			throw new ApiRequestError(axiosError.message || "Unknown error");
		}
	}
}
