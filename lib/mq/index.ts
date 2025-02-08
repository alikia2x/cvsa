import { Queue } from "bullmq";

const MainQueue = new Queue("cvsa");

export default MainQueue;