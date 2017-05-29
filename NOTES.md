Useful notes
==================

As I am using domino datalab to "parallelize" my code, there are few valueble notes about the api. first, they take snapshots of your code every time you run some simulatin. So, when you sync your folder with Domino service, sometimes they download the current state of your files to your local machine and append '-theirs.' to the file. So, if you want to delete them all, run:

    find . -name "*-theirs.*" -type f -delete
