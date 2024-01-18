clear
close all
% folder='/home/adminit/RL_VerticalAxisTurbine/Carousel/2023_BC/bc002/raw/20230816/';
folder='C:\Users\PIVUSER\Desktop\RL_VerticalAxisTurbine\Carousel\2023_BC\bc006\raw\20231201\';
% addpath(genpath('/home/adminit/Documents/MATLAB/fuf'))
% addpath(genpath('/home/adminit/Documents/MATLAB/app_motioncontrol'))
% addpath(genpath('/home/adminit/Documents/MATLAB/src_motioncontrol'))
is_training=false;
ms=8;
mpt=1;
[res, param]=process_ni(ms,mpt,folder,is_training);
load(append(folder,"Cp_phavg.mat"))

[~,locsp] = findpeaks(transpose(res.phase));                                 % Find Maxima & Indices
[~,locsv] = findpeaks(-transpose(res.phase));                                % Find Minima & Indices
pkidx = sort([locsv locsp(locsp > locsv(1))]);             % Edit & Sort Indices
a=1;
for k = 1:2:numel(pkidx)-1
    idxrng = pkidx(k):pkidx(k+1);                           % Index Range For Each Segment
    phase_vector{a} = transpose(res.phase(idxrng));         % Corresponding phase Vector
    Cp_vector{a} = transpose(res.Cp(idxrng));               % Corresponding Cp Vector
    pitch_vector{a} = transpose(rad2deg(res.pitch(idxrng))); 
    alpha_vector{a}=atand(sind(phase_vector{a})./(param.lambda+cosd(phase_vector{a})));% Corresponding pitch Vector
    if is_training
        reward_vector{a}=transpose(res.reward(idxrng));
    end
    a=a+1;
end
param.Cp
meanCp=[];
totalreward=[];
figure;

c=gray(numel(phase_vector)); %colormap
c = flipud(c);

hold on
for k = 1:numel(phase_vector)
    plot(phase_vector{k}, Cp_vector{k},'color',c(k,:))                % plot Results
    meanCp(end+1) = mean(Cp_vector{k});
    if is_training
        totalreward(end+1)=sum(reward_vector{k});
    end
end
xlim([0 360]);
xlabel("phase");
ylabel("Cp");


[max_Cp argmax_Cp] = max(meanCp);
[max_reward argmax_reward] = max(totalreward);
disp(['Best Cp : ' num2str(max_Cp)  ' on rotation ' num2str(argmax_Cp)])
meanCp(argmax_reward);
plot(0:359,Cp_phavg.phavg,'color','#d8b365',linewidth=2)
plot(phase_vector{argmax_Cp}, Cp_vector{argmax_Cp},'color','#5ab4ac',linewidth=2.5)
if is_training
plot(phase_vector{argmax_reward}, Cp_vector{argmax_reward},'color','red',linewidth=2.5)
end
hold off
grid

exportgraphics(gcf,append(folder,sprintf('Cp_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)

figure;
hold on
for k = 1:2:numel(phase_vector)
    plot(phase_vector{k}, (1+Cp_vector{k}-interp1(0:359,Cp_phavg.phavg,phase_vector{k}))/2,'color',c(k,:))                % plot Results
end
plot(phase_vector{argmax_Cp}, (1+Cp_vector{argmax_Cp}-interp1(0:359,Cp_phavg.phavg,phase_vector{argmax_Cp}))/2,'color','#5ab4ac',linewidth=2.5)
if is_training
plot(phase_vector{argmax_reward},(1+ Cp_vector{argmax_reward}-interp1(0:359,Cp_phavg.phavg,phase_vector{argmax_reward}))/2,'color','red',linewidth=2.5)
end
hold off
grid





% % % % % % % % % % % % % % % % % 
d=readtable('pitch_optim');
figure;
hold on
for k = 1:numel(phase_vector)
    plot(phase_vector{k}, pitch_vector{k},'color',c(k,:))                % plot Results
end
xlim([0 360]);
xlabel("phase");
ylabel("pitch");
plot(phase_vector{argmax_Cp}, pitch_vector{argmax_Cp},'color','#5ab4ac',linewidth=2.5)
if is_training
plot(phase_vector{argmax_reward},pitch_vector{argmax_reward},'color','red',linewidth=2.5);
end
% plot(d.Var1,d.Var2,"Color",'green')
hold off
grid
exportgraphics(gcf,append(folder,sprintf('pitch_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)

figure;
hold on
for k = 1:numel(phase_vector)
    plot(phase_vector{k}, pitch_vector{k}+alpha_vector{k},'color',c(k,:))                % plot Results
end
xlim([0 360]);
xlabel("phase");
ylabel("pitch");
plot(phase_vector{argmax_Cp}, alpha_vector{argmax_Cp}+pitch_vector{argmax_Cp},'color','#5ab4ac',linewidth=2.5)
plot(phase_vector{argmax_Cp},alpha_vector{argmax_Cp},'color','#d8b365',linewidth=2)
hold off
grid
exportgraphics(gcf,append(folder,sprintf('alpha_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)
% 
if is_training
    figure;
    hold on
    for k = 1:numel(phase_vector)
        plot(phase_vector{k}, reward_vector{k},'color',c(k,:));                % plot Results
    end
    plot(phase_vector{argmax_Cp},reward_vector{argmax_Cp},'color','#5ab4ac',linewidth=2.5);
    % plot(phase_vector{argmax_Cp},5*(Cp_vector{argmax_Cp}-interp1(0:359,Cp_phavg.phavg,phase_vector{argmax_Cp},'linear',0)),'color','blue');
    plot(phase_vector{argmax_reward},reward_vector{argmax_reward},'color','red',linewidth=2.5);
    exportgraphics(gcf,append(folder,sprintf('reward_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)
end

if ~ is_training
    figure;
    subplot(3,1,1)
    plot(res.phavg_list,res.pitch_phavg,linewidth=2,color='k')
    grid("on")
    xlim([0 360]);
    ylabel("pitch")

    subplot(3,1,2)
    hold on
    plot(res.phavg_list,atand(sind(res.phavg_list)./(param.lambda+cosd(res.phavg_list))),'color','#BDBDBD',linewidth=1.5)
    plot(res.phavg_list,res.pitch_phavg+atand(sind(res.phavg_list)./(param.lambda+cosd(res.phavg_list))),linewidth=2,color='k')
    grid("on")
    xlim([0 360]);
    hold off
    ylabel("effective AOA")
    
    subplot(3,1,3)
    hold on
    plot(0:359,Cp_phavg.phavg,'color','#BDBDBD',linewidth=1.5)
    plot(res.phavg_list,res.Cp_phavg,linewidth=2,color='k')
    
    grid("on")
    xlim([0 360]);
    hold off
    xlabel("phase");
    ylabel("$C_p$", 'Interpreter', 'latex');
    exportgraphics(gcf,append(folder,sprintf('policy_results_ms%03dmpt%03d.png',ms,mpt)),'Resolution',300)

end