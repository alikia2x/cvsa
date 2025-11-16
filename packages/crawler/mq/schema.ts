export class WorkerError extends Error {
	public service?: string;
	public codePath?: string;
	public rawError: Error;
	constructor(rawError: Error, service?: string, codePath?: string) {
		super(rawError.message);
		this.name = "WorkerFailure";
		this.codePath = codePath;
		this.service = service;
		this.rawError = rawError;
	}
}

export interface GetVideoInfoJobData {
	aid: number;
	insertSongs?: boolean;
	uid?: string;
}
