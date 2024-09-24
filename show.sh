ROOT=analysis/scan
for name in pass undo;
do
    splitdir=$ROOT/$name.filelist.split
    # do only when the splitdir exists
    if [ -d $splitdir ]; then
        cat $splitdir/* > $ROOT/$name.filelist
        rm -r $splitdir
        wc -l $ROOT/$name.filelist
    fi
    
done 