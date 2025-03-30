import { createFactory } from 'hono/factory'

const factory = createFactory();

export const createHandlers = factory.createHandlers;