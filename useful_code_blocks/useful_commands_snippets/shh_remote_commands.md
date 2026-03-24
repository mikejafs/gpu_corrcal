ACCESSING TORONTO TEST CLUSTER

    ssh mikejafs@ctc.chord-observatory.ca

$\Rightarrow$ Then ssh into one of the two main test systems (either cx66 or cx77)

For changing persmissions

- `chmod 700 ~`

For verifying permissions

- `chmod 700 ~`

    - Note that when I ran this I got the following:
    - drwxr-xr-x 5 mikejafs mikejafs 4096 Mar  2 16:53 /home/mikejafs
- 

    What drwxr-xr-x Means

    Break it apart:

    d  rwx  r-x  r-x
    │   │    │    │
    │   │    │    └── others (everyone else on machine)
    │   │    └────── group
    │   └────────── owner (you)
    └────────────── directory

    So:

    You (owner) → rwx → full access

    Group → r-x → can read + enter

    Others → r-x → can read + enter

    That means:

    ⚠️ Other users can:

    List files in your home directory

    Enter your home directory

    Read files that are world-readable

    They cannot modify them — but they can see them.

Checking the repo size

- `cd yourrepo`
- `du -sh .`


Checking disk size

-  `df -h`

For checking current usage

- `top`

For checking if the GPUs are free

- `nvidia-smi`

For genereating ssh key (specifically useful for having to generate a new ssh key on the cluster)

-  `ssh-keygen -t ed25519 -C "cluster"`

For testing what CPU hardware is being used

- `lscpu`

For processor model only

- `cat /proc/cpuinfo | grep "model name" | head -1`

For total core count

-  `nproc`
-  

Note that if installing packages into a conda invironment, should really be using the command:

-  `python -m pip install blah` not just `pip install`

  

