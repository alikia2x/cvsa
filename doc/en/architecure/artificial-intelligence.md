# Artificial Intelligence

CVSA's automated workflow relies heavily on artificial intelligence for information extraction and classification.

The AI ​​systems we currently use are:

### The Filter

Located at `/filter/` under project root dir, it classifies a video in the
[category 30](../about/scope-of-inclusion.md#category-30) into the following categories:

* 0: Not related to Chinese vocal synthesis
* 1: A original song with Chinese vocal synthesis
* 2: A cover/remix song with Chinese vocal synthesis

### The Predictor

Located at `/pred/`under the project root dir, it predicts the future views of a video. This is a regression model that takes historical view trends of a video, other contextual information (such as the current time), and future time points to be predicted as feature inputs, and outputs the increment in the video's view count from "now" to the specified future time point.
