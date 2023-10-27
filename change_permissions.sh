export USER2=jean-pierre.falet
setfacl -Rdm user:${USER}:rwx  $SCRATCH/shared/delta_ai
setfacl -Rdm user:${USER2}:rwx $SCRATCH/shared/delta_ai
setfacl -Rm  user:${USER2}:rwx $SCRATCH/shared/delta_ai
setfacl -m   user:${USER2}:x   $SCRATCH/shared/delta_ai
setfacl -m   user:${USER2}:x   $SCRATCH/shared
setfacl -m   user:${USER2}:x   $SCRATCH