
% analyse simulations for varying Node number & processes

clear;

load results_Nodes_Nprocess_15March;

average_error_MT = mean(error_MT_All,[3,4]);
average_error_LS = mean(error_LS_All,[3,4]);
average_error_LS2 = mean(error_LS2_All,[3,4]);

figure;
hold on;
%plot(N_graph_size_set,average_error_MT(:,2:5),'o-','linewidth',2,'MarkerSize',10);
plot(N_graph_size_set,average_error_MT(:,2),'ro-','linewidth',1,'MarkerSize',10);
plot(N_graph_size_set,average_error_MT(:,3),'bs-','linewidth',1,'MarkerSize',10);
plot(N_graph_size_set,average_error_MT(:,4),'k^-','linewidth',1,'MarkerSize',10);
plot(N_graph_size_set,average_error_MT(:,5),'gv-','linewidth',1,'MarkerSize',10);
legend('n = 1','n = 3','n = 5','n = 7');

%plot(N_graph_size_set,average_error_LS2(:,2:5),'s--','linewidth',2,'MarkerSize',10);
p = plot(N_graph_size_set,average_error_LS2(:,2),'ro-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
p = plot(N_graph_size_set,average_error_LS2(:,3),'bs-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
p = plot(N_graph_size_set,average_error_LS2(:,4),'k^-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
p = plot(N_graph_size_set,average_error_LS2(:,5),'gv-.','linewidth',1,'MarkerSize',10);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

grid on; box on;
ylim([0 100]);
xlabel('# Nodes'); ylabel('Average relative error %');
set(gca, 'YScale', 'log');
fontsize(scale=1.5);

% figure;
% hold on;
% plot(N_graph_size_set,average_error_LS,'o-');
% plot(N_graph_size_set,average_error_LS2,'s--');
% grid on;
% ylim([0 100]);
% xlabel('Nodes'); ylabel('Error');
% set(gca, 'YScale', 'log');
