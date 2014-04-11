function ekf_train_test
% init input data
[X,T] = simpleseries_dataset;
% argument don't used
ekf_batch_proceed('trainlm', 0.01, 0.1, 0.2, false, false);
% batch training
ekf_batch_proceed('trainkf', 0.01, 0.1, 0.2, false, false);
% varying P
ekf_batch_proceed('trainkf', 0.001, 0.1, 0.2, false, false);
% varying R
ekf_batch_proceed('trainkf', 0.01, 1, 0.2, false, false);
% varying Q
ekf_batch_proceed('trainkf', 0.01, 1, 0.0001, false, false);
% % incremental training
% ekf_batch_proceed('trainkf', 0.01, 0.1, 0.2, true);
% % varying P
% ekf_batch_proceed('trainkf', 0.001, 0.1, 0.2, true);
% % varying R
% ekf_batch_proceed('trainkf', 0.01, 1, 0.2, true);
% % varying Q
% ekf_batch_proceed('trainkf', 0.01, 1, 0.0001, true);

    function ekf_batch_proceed(func, e, nu, q, isIncremental, toDiagonal)
        net = timedelaynet(1:2, 10);
        if nargin == 6
            net.userdata.e  = e;
            net.userdata.nu = nu;
            net.userdata.q  = q;
            net.userdata.norm_max = 1e3;
            net.userdata.isIncremental = isIncremental;
            net.userdata.toDiagonal = toDiagonal;
        end
        % prepare input data
        net = configure(net,X,T);
        [Xs,Xi,Ai,Ts] = preparets(net,X,T);
        % not divide train set
        net.divideFcn = '';
        % lets target data being accessible during training
        net.userdata.Ts = Ts;
        % init epochs
        net.userdata.epochs = 0;
        % disable window
        net.trainParam.showWindow = false;
        % start train
        if ~isIncremental
            % set our train function for EKF
            net.trainFcn = func; % 'trainkf_n';
            [net,tr] = train(net,Xs,Ts,Xi,Ai);
        else
            % set our train function for EKF
            net.adaptFcn = func; % 'trainkf_n';
            [net,tr] = adapt(net,Xs,Ts,Xi,Ai);
        end
        fprintf('                                                         \n');
        fprintf('\tnumber of epochs  = %d\n', length(tr.epoch));
        [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
        fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
        fprintf('                                                         \n');
        %     plot(cell2mat(Xs),cell2mat(Ts),'o',cell2mat(Xs),cell2mat(Y),'x');
        plot((1:length(Ts)), cell2mat(Ts), '- .k', (1:length(Ts)) ,cell2mat(Y) ,'- .b');
    end
end
