clear
close all
folder='C:\Users\PIVUSER\Desktop\RL_VerticalAxisTurbine\Carousel\2023_BC\bc002\raw\20230721\';

load(append(folder,"Cp_phavg.mat"))
ms=4;
mpt=3;
[res1, param1]=process_ni(ms,mpt,folder);
% res2=process_ni(2,2);
% pitch_12=[res1.pitch; res2.pitch];
% phase_12=[res1.phase; res2.phase];
% Cp_12=[res1.Cp; res2.Cp];
% 
% t=transpose(phase_12);
% % y=transpose(rad2deg(pitch_12));
% y=transpose(Cp_12);
t=transpose(res1.phase);
% y=transpose(rad2deg(res1.pitch));
y=transpose(res1.Cp);
[~,locsp] = findpeaks(t);                                 % Find Maxima & Indices
[~,locsv] = findpeaks(-t);                                % Find Minima & Indices
pkidx = sort([locsv locsp(locsp > locsv(1))]);             % Edit & Sort Indices

for k = 1:2:numel(pkidx)-1
    idxrng = pkidx(k):pkidx(k+1);                           % Index Range For Each Segment
    tv{k} = t(idxrng);                                      % Corrseponding Time Vector
    Asc{k} = y(idxrng);                                   % Corresponding Signal Segment Vector
end

meanCp=[];
figure;
c=gray(numel(tv));
c = flipud(c);
% plot(t, y)                                                  % Plot Original Signal
hold on
for k = 1:numel(tv)
    plot(tv{k}, Asc{k},'color',c(k,:))                % plot Results
    meanCp(end+1) = mean(Asc{k});
end
xlim([0 360]);
xlabel("phase");
ylabel("Cp");
[max_Cp argmax_Cp] = max(meanCp)
plot(tv{argmax_Cp}, Asc{argmax_Cp},'color','red')
plot(0:359,Cp_phavg.phavg)
hold off
grid

exportgraphics(gcf,append(folder,sprintf('Cp_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)
% % % % % % % % % % % % % % % % % 

t=transpose(res1.phase);
y=transpose(rad2deg(res1.pitch));

[pks,locsp] = findpeaks(t);                                 % Find Maxima & Indices
[vls,locsv] = findpeaks(-t);                                % Find Minima & Indices
pkidx = sort([locsv locsp(locsp > locsv(1))]);              % Edit & Sort Indices
for k = 1:2:numel(pkidx)-1
    idxrng = pkidx(k):pkidx(k+1);                           % Index Range For Each Segment
    tv{k} = t(idxrng);                                      % Corrseponding Time Vector
    Asc{k} = y(idxrng);                                     % Corresponding Signal Segment Vector
end

figure;
c=gray(numel(tv));
c = flipud(c);
% plot(t, y)                                                  % Plot Original Signal
hold on
for k = 1:numel(tv)
    plot(tv{k}, Asc{k},'color',c(k,:))                % plot Results

end
xlim([0 360]);
xlabel("phase");
ylabel("pitch");
plot(tv{argmax_Cp}, Asc{argmax_Cp},'color','red')

hold off
grid
exportgraphics(gcf,append(folder,sprintf('pitch_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)