export const log = (value: number, base: number = 10) => Math.log(value) / Math.log(base);

export const truncate = (num: number, min: number, max: number) => Math.max(min, Math.min(num, max));
