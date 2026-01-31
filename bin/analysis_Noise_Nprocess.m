
% analyse simulations for varying Node number & processes

clear;

load results_Noise_Nprocess_15March;

average_error_MT = mean(error_MT_All,[3,4]);
average_error_LS = mean(error_LS_All,[3,4]);
average_error_LS2 = mean(error_LS2_All,[3,4]);

SNR_list = 10*log10(1./noise_sigma_All);

figure;
hold on;
%plot(SNR_list,average_error_MT,'o-','linewidth',1,'MarkerSize',10);
plot(SNR_list,average_error_MT(:,2),'ro-','linewidth',1,'MarkerSize',10);
plot(SNR_list,average_error_MT(:,3),'bs-','linewidth',1,'MarkerSize',10);
plot(SNR_list,average_error_MT(:,4),'k^-','linewidth',1,'MarkerSize',10);
plot(SNR_list,average_error_MT(:,5),'gv-','linewidth',1,'MarkerSize',10);
legend('n = 1','n = 3','n = 5','n = 7');

%plot(SNR_list,average_error_LS2,'s--','linewidth',1,'MarkerSize',10);
p = plot(SNR_list,average_error_LS2(:,2),'ro-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
p = plot(SNR_list,average_error_LS2(:,3),'bs-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
p = plot(SNR_list,average_error_LS2(:,4),'k^-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
p = plot(SNR_list,average_error_LS2(:,5),'gv-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
grid on; box on;
ylim([0 100]);
set(gca, 'YScale', 'log');
xlabel('SNR'); ylabel('Average relative error %');
fontsize(scale=1.5);

% figure;
% hold on;
% plot(SNR_list,average_error_LS,'o-');
% plot(SNR_list,average_error_LS2,'s--');
% grid on;
% xlabel('SNR'); ylabel('Error');
% set(gca, 'XScale', 'log');
% ylim([0 100]);
%set(gca, 'YScale', 'log');
