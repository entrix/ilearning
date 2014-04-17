function ekf_train_test
% init input data
[X,T] = simpleseries_dataset;
% argument don't used
ekf_batch_proceed('trainlm', 0.01, 0.1, 0.2, false, false);
% % batch training
ekf_batch_proceed('trainekf', 0.01, 0.1, 0.2, false, false);
% % adapt training
ekf_batch_proceed('adaptekf', 0.01, 0.1, 0.2, true, false);
% % adapt multistream training
ekf_batch_proceed('adaptekf_ms', 0.01, 0.1, 0.2, true, false);
% unscented adapt training
ekf_batch_proceed('adaptukf', 0.0001, 0.1, 20000000, true, false);
% varying P
% ekf_batch_proceed('trainekf', 0.001, 0.1, 0.2, false, false);
% varying R
% ekf_batch_proceed('trainekf', 0.01, 1, 0.2, false, false);
% varying Q
% ekf_batch_proceed('trainekf', 0.01, 1, 0.0001, false, false);
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
        if strcmp(func, 'trainlm') == 1
          fprintf(strcat('\trun function: %s\n'), func);
          t = cputime;
          [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
          fprintf('\tElapsed time is %f seconds.\n', 1000*(cputime-t));
          fprintf('                                                         \n');
          fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
        elseif ~isIncremental
            fprintf(strcat('\run function %s\n'), func);
            % set our train function for EKF
            net.trainFcn = func; % 'trainkf_n'; 
            for i = 1:10
              [net,tr] = train(net,Xs,Ts,Xi,Ai);
              fprintf('                                                         \n');
              fprintf('\tnumber of epoch  = %d\n', i);
              t = cputime;
              [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
              fprintf('\tElapsed time is %f seconds.\n', 1000*(cputime-t));
              fprintf('                                                         \n');
              fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
              fprintf('                                                         \n');
            end
        elseif strcmp(func,'adaptekf_ms') == 1
            fprintf(strcat('\trun function %s\n'), func);
            % set our train function for EKF
            net.adaptFcn = func; % 'trainkf_n';
            net.userdata.l = 11;
            net.userdata.g = 10;
%             [net] = adapt_2(net,Xs,Ts,Xi,Ai);
            for i = 1:10
              [net,Y,E,Xf,Af,tr,WB_list]=adapt_ms(net,Xs,Ts,Xi,Ai);
              fprintf('                                                         \n');
              fprintf('\tnumber of epoch  = %d\n', i);
              t = cputime;
              [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
              fprintf('\tElapsed time is %f seconds.\n', 1000*(cputime-t));
              fprintf('                                                         \n');
              fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
              fprintf('                                                         \n');
            end
        else 
            fprintf(strcat('\trun function %s\n'), func);
            % set our train function for EKF
            net.adaptFcn = func; % 'trainkf_n';
%             net.userdata.L = 98;
            net.userdata.alpha = 0.0001;
            net.userdata.beta  = 2;
%             net.userdata.k  = 3 - net.userdata.L;
            net.userdata.lambda_RLS = 0.5;
%             [net] = adapt_2(net,Xs,Ts,Xi,Ai);
            for i = 1:10
              WB = getwb(net);
              net = adapt_2(net,Xs(1:10),Ts(1:10),Xi,Ai);
              fprintf('                                                         \n');
              fprintf('\tnumber of epoch  = %d\n', i);
              t = cputime;
              [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
              fprintf('\tElapsed time is %f seconds.\n', 1000*(cputime-t));
              fprintf('                                                         \n');
              fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
              fprintf('                                                         \n');
            end
          end
%         fprintf('                                                         \n');
        %     plot(cell2mat(Xs),cell2mat(Ts),'o',cell2mat(Xs),cell2mat(Y),'x');
%         plot((1:length(Ts)), cell2mat(Ts), '- .k', (1:length(Ts)) ,cell2mat(Y) ,'- .b');
    end
end
