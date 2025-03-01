import logger from "lib/log/logger.ts";

logger.error(Error("test error"), "test service");
logger.debug(`some string`);
logger.debug(`hello`, "servicename");
logger.debug(`hello`, "servicename", "codepath.ts");
logger.log("something");
logger.log("foo", "service");
logger.log("foo", "db", "insert.ts");
logger.warn("warn");
logger.error("error");
logger.verbose("error");
