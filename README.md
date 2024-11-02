# Mission: Estimating fish activity in videos

Now we aim at realizing the auto-feeding system that automatically judges when it should change the feeding configuration. This auto-feeding system changes its feeding configuration based on the output of the `fish activity assessment program`.

In this mission, please implement codes of this `fish activity assessment program` whose input is video and output is `fish activity`.
The outputs to be submitted in this task are as follows,

- Codes of `fish activity assessment program`.

  - Input data are videos in `data` directory, and output data are `fish activity` for each input video.
  - Any form of `fish activity` is acceptable, such as fish activity classes (0: low, 1: middle, 2: high) or fish activity score (0-100: higher is active), etc...
  - Please consider what format might be useful for auto-feeding system to change feeding configuration.

- A short report that explains your approach for solving this task and the reason why you took this approach.

In the `data` directory, there are 10 videos of a fish pen during feeding time. Please use these videos for evaluating your approach.
Since they do not have an activity label or activity score, please define them and give this information to the videos by yourself.
(Basically, it is said that fish move fast when they are eating feed and move slowly when they are not eating feed.)

## Note

- Write code in Python.
- Specify a development environment to check code execution or use package managers, e.g. docker, poetry, and so on.
- We will evaluate code clarity and maintainability. Keep the following points in mind:
  - Product development is performed by multiple people.
  - The code for production is reviewed.
  - The code will be changed after release.
