Useful notes
==================

As I am using Domino data lab to "parallelize" my code, there are few valuable notes about the API. First, they take snapshots of your code every time you run some simulation. So, when you sync your folder with Domino service, sometimes they download the current state of your files to your local machine and append '-theirs.' to the file name. So, if you want to delete them all, run:

    find . -name "*-theirs.*" -type f -delete
