# VideoTagsQueue

### Jobs

The VideoTagsQueue contains two jobs: `getVideoTags`and `getVideosTags`. The former is used to fetch the tags of a
video, and the latter is responsible for scheduling the former.

### Return value

The return values across two jobs follows the following table:

<table><thead><tr><th width="168">Return Value</th><th>Description</th></tr></thead><tbody><tr><td>0</td><td>In <code>getVideoTags</code>: the tags was successfully fetched<br>In <code>getVideosTags</code>: all null-tags videos have a corresponding job successfully queued.</td></tr><tr><td>1</td><td>Used in <code>getVideoTags</code>: occured <code>fetch</code>error during the job</td></tr><tr><td>2</td><td>Used in <code>getVideoTags</code>: we've reached the rate limit set in NetScheduler</td></tr><tr><td>3</td><td>Used in <code>getVideoTags</code>: did't provide aid in the job data</td></tr><tr><td>4</td><td>Used in<code>getVideosTags</code>: There's no video with NULL as `tags`</td></tr><tr><td>1xx</td><td>Used in<code>getVideosTags</code>:  the number of tasks in the queue has exceeded the limit, thus <code>getVideosTags</code> stops adding tasks. <code>xx</code> is the number of jobs added to the queue during execution.</td></tr></tbody></table>
