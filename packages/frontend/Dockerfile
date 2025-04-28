FROM oven/bun AS bun-builder

WORKDIR /app

COPY package.json bun.lock tsconfig.json astro.config.mjs ./

RUN bun install

COPY . .

RUN bun run build

FROM node:lts-alpine

WORKDIR /app

COPY --from=bun-builder /app/dist ./dist
COPY --from=bun-builder /app/node_modules ./node_modules
COPY --from=bun-builder /app/package.json ./package.json


ENV HOST=0.0.0.0
ENV PORT=4321

EXPOSE 4321

CMD ["node", "dist/server/entry.mjs"]