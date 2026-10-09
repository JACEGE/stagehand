---
"@browserbasehq/stagehand": minor
---

add `experimentalNavigationGraph.replayWithHealing()`, which retries a failed navigation step through `act()` inference once and stores the healed action when the expected state is reached.
