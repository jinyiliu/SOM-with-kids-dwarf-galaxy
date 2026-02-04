# Create a directory and then run this script inside it to
# download the KiDS DR4 BASIC_RANDOMS data.
wget --user=KiDS_Collaboration --ask-password -r -np -nc -R "index.html*" --no-directories "http://cuillin.roe.ac.uk/~cech/KiDS/KiDS-1000/BASIC_RANDOMS/"