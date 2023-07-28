clear
close all
folder='C:\Users\PIVUSER\Desktop\RL_VerticalAxisTurbine\Carousel\2023_BC\bc002\raw\20230727\';

load(append(folder,"Cp_phavg.mat"))
ms=2;
mpt=4;
[res, param]=process_ni(ms,mpt,folder);

[~,locsp] = findpeaks(transpose(res.phase));                                 % Find Maxima & Indices
[~,locsv] = findpeaks(-transpose(res.phase));                                % Find Minima & Indices
pkidx = sort([locsv locsp(locsp > locsv(1))]);             % Edit & Sort Indices

for k = 1:2:numel(pkidx)-1
    idxrng = pkidx(k):pkidx(k+1);                           % Index Range For Each Segment
    phase_vector{k} = transpose(res.phase(idxrng));         % Corrseponding phase Vector
    Cp_vector{k} = transpose(res.Cp(idxrng));               % Corresponding Cp Vector
    pitch_vector{k} = transpose(res.pitch(idxrng));                            % Corresponding pitch Vector
end

meanCp=[];
figure;

c=gray(numel(phase_vector)); %colormap
c = flipud(c);

hold on
for k = 1:numel(phase_vector)
    plot(phase_vector{k}, Cp_vector{k},'color',c(k,:))                % plot Results
    meanCp(end+1) = mean(Cp_vector{k});
    
end
xlim([0 360]);
xlabel("phase");
ylabel("Cp");

[max_Cp argmax_Cp] = max(meanCp)

plot(0:359,Cp_phavg.phavg,'color','yellow',linewidth=2)
plot(phase_vector{argmax_Cp}, Cp_vector{argmax_Cp},'color','red',linewidth=2)

hold off
grid

exportgraphics(gcf,append(folder,sprintf('Cp_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)
% % % % % % % % % % % % % % % % % 

figure;
hold on
for k = 1:numel(phase_vector)
    plot(phase_vector{k}, pitch_vector{k},'color',c(k,:))                % plot Results
end
xlim([0 360]);
xlabel("phase");
ylabel("pitch");
plot(phase_vector{argmax_Cp}, pitch_vector{argmax_Cp},'color','red',linewidth=2)

hold off
grid
exportgraphics(gcf,append(folder,sprintf('pitch_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)

alpha=atand(sind(res.phase)./(param.lambda+cosd(res.phase)));
figure;
plot(res.phase,alpha)
plot(res.phase,alpha+res.pitch)