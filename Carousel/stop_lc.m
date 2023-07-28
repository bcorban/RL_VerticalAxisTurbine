lc.stopLC()
volts_ni = lc.data;
t_ni = lc.time;

file_folder = 'C:\Users\PIVUSER\Desktop\RL_VerticalAxisTurbine\Carousel\2023_BC\bc002\raw\20230727';
ms = 2;
% mpt: just count the number of file ms00*mpt* and add +1.
mpt = fix(length(dir(fullfile(file_folder,sprintf("ms%03d*", ms))))/3) + 1;
file_name = sprintf("ms%03dmpt%03d_3.mat", ms, mpt);

file_path = fullfile(file_folder, file_name);
% Check if folder exists, otherwise create it.
if exist(file_folder,'dir') == 0
    mkdir(file_folder)
end
% Save into a .mat file
save(file_path, 'volts_ni', 't_ni');
