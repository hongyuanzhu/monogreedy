function [acc, tpi] = accTime(archiN, convoStages, fullStages)

%% write archi
fid = fopen('archi.txt','w');
fprintf(fid,'%d\n',convoStages);
fprintf(fid,'%d\n',fullStages);
for i = 1:convoStages
    fprintf(fid,'%d\n',archiN(i));
end
for i = convoStages+1:convoStages+fullStages
    fprintf(fid,'%d\n',archiN(i));
end
fclose(fid);

%% call python
envtemp = getenv('LD_LIBRARY_PATH'); setenv('LD_LIBRARY_PATH','');
system('python MLaccTime.py');
setenv('LD_LIBRARY_PATH',envtemp);

aT = textread('accTime.txt','%f');

acc = aT(1);
tpi = aT(2);