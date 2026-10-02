# General TODOs

*Note that this file should be more for todos at a high level like "work on this experiment todos" or "add results to this", begin thinking about such and such etc... Not concrete plans at a granular level partaining to a particular experiment or sub file*

- [2026-03-24](#2026-03-24)
- [2026-03-04](#2026-03-04) $\rightarrow$ ***From backlog thesis project roughwork*** 
- [2026-04-15](#2026-04-15)
- [2026-05-14](#2026-05-14)
- [2026-10-01](#2026-10-01)


## 2026-03-24

~~- Fill out the rest of this repo with relevant active files and begin using~~
  - ~~probably just using the experiments folder for now~~

~~- clean up other files from Thesis project work so we can safely and cleanly stop using stuff there~~

~~- Get a hang of using pip install -e for installing this gpu_corrcal in an environment~~

~~- AS SOON AS WE FEEL THE REPO IS IS 'WORKING ORDER':~~
  ~~- *BEGIN WORKING TO UNDERSTAND THE WARPED REDUCTION KERNEL* $\rightarrow$ abuse the experiments hierarchy to run smaller tests that enable a deeper understanding of the constituent parts~~

<br>
<br>

***From old CUDA programming work log:***


## 2026-03-04

***
      NEED TO START PLANNING WHEN TO MAKE MATERIAL PROGRESS ON BANDPASS PROJECT
      -> I THINK THE GOAL WILL BE TO CLEAN UP AND ORGANIZE THE CURRENT GPU WORK and FIGURE OUT HOW THE RED KERNEL IS WORKING
      -> THEN CAN START TRYING TO BUILD A FEW COVARIANCE MATRICES AND WRITE OUT THE FORMALISM BETTER FOR THIS STAGE OF THE BANDPASS PROJECT
        -> MAKE A DEDICATED BANDPASS REPO AND START COMMITTING CODE
      

      -> GETTING THE CUPY CODE RUNNING AND CLEANED UP WITH A BIT MORE DOCUMENTATION CAN HAPPEN INTERMIDENTLY
***

~~- ***Priority:*** REORGANIZE ALL CODE AND BETTER CREATE ACCOMPANYING GITHUB REPOS FOR EACH DIRECTORY, FIGURE OUT HOW TO EXPORT ENVS AS PART OF GITHGUB BACKUP -- SEE CHATGPT CONVO~~

- keep picking apart red kern code using chatgpt, while reading explicit technical blogs
  - Begin making smaller test cases that utilize concepts individually (essentially learning hands on)
  - during organization process above, need to make gpu learning repo and begin making mroe thorough gpu notes on important/relevant concepts


### Next (backlog -- start prioritizing this to get out of the way, do once clean up current tests and understand red kern, but before serious bandpass work):
1. Double check expected sizings for bandpass cal covariance sizes and include these timings in notes above
2. ~~Make documented notes in corrcal notes overleaf with some plots of all of the above tests — move things to well documented and labelled folders with readmes and file descriptions~~ 
   ~~1. **$\Rightarrow$ Simply add current results to the results folder with some description of what we did to achieve these plots.**~~


## 2026-04-15
- Continue reading through the gpu_techniques file to better understand the kernel $\rightarrow$ we're almost there....
- double check sizings for bandpass cal cov as soon as we begin this to be sure it will work with current implementation
- Get going on bandpass (as soon as remotely confortable if asked to explain the gpu kernel):
  - Create dedicated bandpass repo and start committing code to and experiments folder
    - likely just begin trying to get a covariance written down
- ***GET CUPY SEFL-CONTAINED LIBRARY FINISHED AND DONE WITH SO CAN POST TO THE WEBSITE***
- **Code up relevant tests for correctness and timing comparisons relative to the hardcoded, bobby's, and cupy version $\Rightarrow$ Then fully ready to begin coding other corrcal inversion functions**
- *URGENT:* To the above point, essentially following the TODOs listed within the warp reduction experiment as the main priority so we can get moving to writing the next functions asap 


## 2026-05-14

### Soon (After main GPU work has good progress)
- Double check sizings for bandpass cal cov as soon as we begin this to be sure it will work with current implementation
- Get going on bandpass (as soon as remotely confortable if asked to explain the gpu kernel):
  - Create dedicated bandpass repo and start committing code to and experiments folder
    - likely just begin trying to get a covariance written down
- ***GET CUPY SEFL-CONTAINED LIBRARY FINISHED AND DONE WITH SO CAN POST TO THE WEBSITE***

### Urgent
- **Code up relevant tests for correctness and timing comparisons relative to the hardcoded, bobby's, and cupy version $\Rightarrow$ Then fully ready to begin coding other corrcal inversion functions**
- *URGENT:* To the above point, essentially following the TODOs listed within the warp reduction experiment as the main priority so we can get moving to writing the next functions asap $\Rightarrow$ No need to tie up every loose end, just clean things up so we had some idea of wtf was going on when we were working on that stuff

## 2026-10-01
- Finish remaining corrcal functions and kernels and add to src code
  - Remaining:
    - multiply vec by mat (JUST NEED TO CHECK BENCHMARKING) 
    - apply gains
    - full likelihood (likely just python side)
    - gradient kernel (+python code)
- take a solid stab at curvature
- start getting bandpass going