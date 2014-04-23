function ekf_train_test
% init input data
[X,T] = simpleseries_dataset;
% argument don't used
ekf_batch_proceed('trainlm', 0.01, 0.1, 0.2, false, false);
% batch ekf training
ekf_batch_proceed('trainekf', 0.001, 1, 0.2, true, false);
% batch ukf training
ekf_batch_proceed('trainukf', 0.001, 0.5, 200, true, false);
% adapt ekf training
ekf_batch_proceed('adaptekf', 0.01, 0.1, 0.2, true, false);
% adapt ekf multistream training
ekf_batch_proceed('adaptekf_ms', 0.01, 0.1, 0.2, true, false);
% adapt ukf training
ekf_batch_proceed('adaptukf', 0.01, 0.5, 200, true, false);

  function ekf_batch_proceed(func, e, nu, q, isIncremental, toDiagonal)
    net = timedelaynet(1:2, 10);
    if nargin == 6
      net.userdata.e  = e;
      net.userdata.nu = nu;
      net.userdata.q  = q;
      net.userdata.norm_max = 1e3;
      % default covariance matrix
      net.userdata.P = [];
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
    net.trainParam.showWindow = true;
    
    % start train
    if strcmp(func, 'trainlm') == 1
      fprintf(strcat('\trun function: %s\n'), func);
      t = cputime;
      [net,tr] = train(net,Xs,Ts,Xi,Ai);
      fprintf('\tElapsed time is %f seconds.\n', (cputime-t));
      [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
      fprintf('                                                         \n');
      fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
    elseif strcmp(func, 'trainekf') == 1
      fprintf(strcat('\trun function %s\n'), func);
      % set our train function for EKF
      net.trainFcn = func; % 'trainkf_n';
      net.userdata.checkIter = true;
      net.userdata.printMatrix = false;
      net.trainParam.epochs = 100;
      for i = 1:10
        net.userdata.loop_n = i;
        t = cputime;
        [net,tr] = train_p(net,Xs,Ts,Xi,Ai);
        fprintf('\tElapsed time is %f seconds.\n', (cputime-t));
        fprintf('                                                         \n');
        fprintf('\tnumber of epoch  = %d\n', i);
        [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
        fprintf('                                                         \n');
        fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
        fprintf('                                                         \n');
      end
    elseif strcmp(func, 'trainukf') == 1
      fprintf(strcat('\trun function %s\n'), func);
      % set our train function for EKF
      net.trainFcn = func; % 'trainkf_n';
      net.userdata.checkIter = true;
      net.userdata.printMatrix = false;
      net.trainParam.epochs = 2000;
      % ukf parameters
      net.userdata.alpha = 1e-3;
      net.userdata.beta  = 2;
      net.userdata.lambda_RLS = 9e-1;
      net.userdata.alpha_RM = 0.2;
      net.userdata.checkIter = true;
      net.trainParam.epochs = 10;
      for i = 1:10
        net.userdata.loop_n = i;
        t = cputime;
        [net,tr] = train_p(net,Xs,Ts,Xi,Ai);
        fprintf('\tElapsed time is %f seconds.\n', (cputime-t));
        fprintf('                                                         \n');
        fprintf('\tnumber of epoch  = %d\n', i);
        [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
        fprintf('                                                         \n');
        fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
        fprintf('                                                         \n');
      end
    elseif strcmp(func,'adaptekf_ms') == 1
      fprintf(strcat('\trun function %s\n'), func);
      % set our train function for EKF
      net.adaptFcn = func;
      net.userdata.l = 10;
      net.userdata.checkIter = true;
      for i = 1:10
        net.userdata.loop_n = i;
        t = cputime;
        net = adapt_x(net,Xs,Ts,Xi,Ai);
        fprintf('\tElapsed time is %f seconds.\n', (cputime-t));
        fprintf('                                                         \n');
        fprintf('\tnumber of epoch  = %d\n', i);
        [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
        fprintf('                                                         \n');
        fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
        fprintf('                                                         \n');
      end
    elseif strcmp(func,'adaptekf') == 1
      fprintf(strcat('\trun function %s\n'), func);
      net.adaptFcn = func;
      net.userdata.checkIter = true;
      net.userdata.printMatrix = true;
      for i = 1:10
        net.userdata.loop_n = i;
        t = cputime;
        net = adapt_x(net,X(randperm(numel(X))),T(randperm(numel(T))),Xi,Ai);
        fprintf('\tElapsed time is %f seconds.\n', (cputime-t));
        fprintf('                                                         \n');
        fprintf('\tnumber of epoch  = %d\n', i);
        fprintf('                                                         \n');
        [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
        fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
        fprintf('                                                         \n');
      end
    elseif strcmp(func,'adaptukf') == 1
      fprintf(strcat('\trun function %s\n'), func);
      % set our train function for EKF
      net.adaptFcn = func;
      net.userdata.alpha = 1e-3;
      net.userdata.beta  = 2;
      net.userdata.lambda_RLS = 9e-1;
      net.userdata.alpha_RM = 0.2;
      net.userdata.checkIter = true;
      net.userdata.printMatrix = true;
      for i = 1:10
        net.userdata.loop_n = i;
        t = cputime;
        net = adapt_x(net,X(randperm(10)),T(randperm(10)),Xi,Ai);
        fprintf('\tElapsed time is %f seconds.\n', (cputime-t));
        fprintf('                                                         \n');
        fprintf('\tnumber of epoch  = %d\n', i);
        fprintf('                                                         \n');
        [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
        fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
        fprintf('                                                         \n');
      end
    end
  end
end
