### Chat with Bobby from 2026-10-28 about merging corrcal code
- ** attend pathfinder calls for corrcal paper running on real data stuff
- ** start loading in pathfinder data and playing around with it
- talk to jim and andre full gpu_corrcal pipeline set up (wait ~~ 1-2 months for bobby to get his side of tings set up)
- ** start thinking about writing corrcal applied to chord data paper and what this would entail and look like
- ! start overleaf docunment for calibrating chord using pathfinder -- add bobby as collaborator



#### current todos
- continue with work for pull request
    - forgot to make a new branch while editing code
    - => Do this asap and make a new dev file, reclone the repo and transfer code to there THEN maek a new branch and add existing code there
    - currently in the middle of reading through code, validating, getting rid of commented or unecessary code
        - currently working through kernels (just starting)
        - all other code looks fine and I'm fairly confident it's calculating the correct thing
    - once kernel code is cleaned up at the top of each file -- make the pull request asap
    - work to read through and understand a bit more of the gpu kernel 
    - just double check how the other kernels work to be reminded of that
    - begin writing up a doc that outlines how they all work and the design patterns/choices that have gone into them
        - this may become a thesis sub-chapter (paired with the corrcal paper) AND IS ESSENTIALLYL NECESSARY FOR BEING ABLE TO GIVE SOME NICE TALKS COMING UP
        - lends nicely for a discussion of how this improvement helps chord and also provides a space for writing about how to merge code with the real-time analysis pipeline
        
